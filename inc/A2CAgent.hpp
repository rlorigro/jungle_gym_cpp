#pragma once

#include <iostream>
#include "Hyperparameters.hpp"
#include "Environment.hpp"
#include "Episode.hpp"
#include "Policy.hpp"
#include <functional>
#include <memory>
#include <filesystem>

using std::filesystem::path;
using std::function;
using std::shared_ptr;
using std::make_shared;


namespace JungleGym{


/**
 * Object which handles the training and testing of the A2C (policy gradient) RL model/algorithm
 */
class A2CAgent {
    shared_ptr<Model> actor;
    shared_ptr<Model> critic;
    Episode episode;
    torch::optim::RMSprop optimizer_actor;
    torch::optim::RMSprop optimizer_critic;
    Hyperparameters hyperparams;

public:
    inline A2CAgent(const Hyperparameters& hyperparams, shared_ptr<Model> actor, shared_ptr<Model> critic);

    inline void train(shared_ptr<const Environment> env);
    inline void test(shared_ptr<const Environment> env);

    /**
     * Additional signature for access to internal model params and optimizers, for use with the A3C algo for param sharing
     * @param env the environment to train on
     * @param f method for A3C parameter syncing. Return true to skip subsequent worker optimize::step(), if desired
     */
    inline void train(
        shared_ptr<const Environment> env,
        const function<bool(shared_ptr<Model> actor, shared_ptr<Model> critic, size_t& e)>& f
        );

    inline void save(const path& output_path) const;
    inline void load(const path& actor_path, const path& critic_path);

};


void A2CAgent::save(const path& output_dir) const{
    if (not std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }

    path actor_path = output_dir / "actor.pt";
    path critic_path = output_dir / "critic.pt";

    torch::save(actor, actor_path);
    torch::save(critic, critic_path);
}


inline void A2CAgent::load(const path& actor_path, const path& critic_path) {
    torch::load(actor, actor_path);
    torch::load(critic, critic_path);
}


A2CAgent::A2CAgent(const Hyperparameters& hyperparams, shared_ptr<Model> actor, shared_ptr<Model> critic):
        actor(actor),
        critic(critic),
        episode(),
        optimizer_actor(actor->parameters(), torch::optim::RMSpropOptions(hyperparams.learn_rate)),
        optimizer_critic(critic->parameters(), torch::optim::RMSpropOptions(hyperparams.learn_rate)),
        hyperparams(hyperparams)
{
    if (!actor) {
        throw std::runtime_error("ERROR: actor pointer is null");
    }
    if (!critic) {
        throw std::runtime_error("ERROR: critic pointer is null");
    }

}


void A2CAgent::train(shared_ptr<const Environment> env) {
    // By default, don't need to do anything with the internals for vanilla A2C. Instead provide null function.
    // Returning false results in default worker-specific optimizer::step().
    auto f_null = [&](shared_ptr<Model> a,shared_ptr<Model> b, size_t c){return false;};

    train(env, f_null);
}


void A2CAgent::train(shared_ptr<const Environment> env, const function<bool(shared_ptr<Model> actor, shared_ptr<Model> critic, size_t& e)>& f){
    if (!env) {
        throw std::runtime_error("ERROR: Environment pointer is null");
    }

    // make a copy of the environment
    shared_ptr<Environment> environment = env->clone();
    environment->reset();

    // Used for deciding if we want to act greedily or sample randomly
    mt19937 generator(1337);
    std::uniform_int_distribution<int64_t> choice_dist(0,4-1);

    size_t e = 0;

    // TODO: un-remove epsilon greedy implementation? make optional?
    float epsilon = 0;

    while (e < hyperparams.n_episodes) {
        episode.clear();

        // exponential decay that terminates at ~0.018
        // epsilon = pow(0.99,(float(e)/float(params.n_episodes)) * float(size_t(eps_norm))));
        std::bernoulli_distribution dist(epsilon);

        for (size_t s=0; s<hyperparams.episode_length; s++) {
            Tensor input = environment->get_observation_space();
            input += 0.0001;

            // Get value prediction (singleton tensor)
            auto value_predict = torch::flatten(critic->forward(input));

            // Get action distribution of the policy (shape of action space)
            auto log_probabilities = torch::flatten(actor->forward(input));
            auto probabilities = torch::exp(log_probabilities);

            int64_t choice;

            if (dist(generator) == 1) {
                choice = choice_dist(generator);
            }
            else {
                // choice = torch::argmax(probabilities).item<int64_t>();
                choice = torch::multinomial(probabilities, 1).item<int64_t>();
            }

            environment->step(choice);

            float reward = environment->get_reward();

            episode.update(log_probabilities, value_predict, choice, reward);

            if (environment->is_terminated() or environment->is_truncated()) {
                environment->reset();
                break;
            }
        }

        auto td_loss = episode.compute_td_loss(hyperparams.gamma, false, true, environment->is_terminated());
        auto entropy_loss = episode.compute_entropy_loss(false, true);

        auto actor_loss = td_loss - hyperparams.lambda*entropy_loss;
        auto critic_loss = 0.5*episode.compute_critic_loss(hyperparams.gamma, false, environment->is_terminated());

        if (not hyperparams.silent and e % 10 == 0) {
            // Print some stats, increment loss using episode, update model if batch_size accumulated
            cerr << std::setprecision(3) << std::left
            << std::setw(7) << e
            << std::setw(4) << episode.get_size()
            << std::setw(10) << "l_entropy" << std::setw(12) << entropy_loss.item<float>()*hyperparams.lambda
            << std::setw(8) << "entropy" << std::setw(12) << entropy_loss.item<float>()/float(episode.get_size())
            << std::setw(5)  << "l_td " << std::setw(12) << td_loss.item<float>()
            << std::setw(9) << "l_critic" << std::setw(12) << critic_loss.item<float>() << '\n';
        }

        actor_loss.backward();
        critic_loss.backward();

        // False by default, true if using A3C parameter sync fn. If using A3C the grads are sent to the master params
        // instead and this worker's model is updated by fetching the master params, so no local step() is needed.
        // In A3C, the episode count (or step count) is maintained at the global level, so our e counter is set by
        // the main thread.
        bool is_worker = f(actor, critic, e);

        if (is_worker) {
            continue;
        }

        e++;

        // Periodically apply the accumulated gradient to the model
        // if (e % std::max(params.batch_size/2,size_t(1)) == 0){
        if (e % hyperparams.batch_size == 0){
            optimizer_critic.step();
            optimizer_critic.zero_grad();
        }

        // Periodically apply the accumulated gradient to the model
        if (e % hyperparams.batch_size == 0){
            optimizer_actor.step();
            optimizer_actor.zero_grad();
        }
    }
}


void A2CAgent::test(shared_ptr<const Environment> env){
    if (!env) {
        throw std::runtime_error("ERROR: Environment pointer is null");
    }

    shared_ptr<Environment> environment = env->clone();

    auto prev_action = environment->get_action_space();

    size_t e = 0;
    std::thread t(std::bind(&Environment::render, environment, false));

    while (e < hyperparams.n_episodes) {
        environment->reset();

        for (size_t s=0; s<hyperparams.episode_length; s++) {
            auto input = environment->get_observation_space().clone();
            input += 0.0001;

            // Get value prediction (singleton tensor)
            auto value_predict = critic->forward(input);

            // Get action distribution of the policy (shape of action space)
            auto log_probabilities = actor->forward(input);
            auto probabilities = torch::exp(log_probabilities);

            cerr << probabilities << '\n';

            int64_t choice = torch::argmax(probabilities).item<int64_t>();
            // choice = torch::multinomial(probabilities, 1).item<int64_t>();

            environment->step(choice);

            if (environment->is_terminated() or environment->is_truncated()) {
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }

        e++;
    }

    t.join();
}


}
