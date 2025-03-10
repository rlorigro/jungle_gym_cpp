#pragma once

#include <iostream>
#include "Hyperparameters.hpp"
#include "Environment.hpp"
#include "Episode.hpp"
#include "Policy.hpp"
#include "memory"

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
    Hyperparameters params;

public:
    inline A2CAgent(const Hyperparameters& params, shared_ptr<Model> actor, shared_ptr<Model> critic);
    inline void train(shared_ptr<const Environment> env);
    inline void test(shared_ptr<const Environment> env);
};


A2CAgent::A2CAgent(const Hyperparameters& params, shared_ptr<Model> actor, shared_ptr<Model> critic):
        actor(actor),
        critic(critic),
        episode(),
        optimizer_actor(actor->parameters(), torch::optim::RMSpropOptions(params.learn_rate)),
        optimizer_critic(critic->parameters(), torch::optim::RMSpropOptions(params.learn_rate)),
        params(params)
{
    if (!actor) {
        throw std::runtime_error("ERROR: actor pointer is null");
    }
    if (!critic) {
        throw std::runtime_error("ERROR: critic pointer is null");
    }

}


void A2CAgent::train(shared_ptr<const Environment> env){
    if (!env) {
        throw std::runtime_error("ERROR: Environment pointer is null");
    }

    // make a copy of the environment
    shared_ptr<Environment> environment = env->clone();

    // Used for deciding if we want to act greedily or sample randomly
    mt19937 generator(1337);
    std::uniform_int_distribution<int64_t> choice_dist(0,4-1);

    size_t e = 0;

    // TODO: un-remove epsilon greedy implementation? make optional?
    float epsilon = 0;

    while (e < params.n_episodes) {
        environment->reset();
        episode.clear();

        // exponential decay that terminates at ~0.018
        // epsilon = pow(0.99,(float(e)/float(params.n_episodes)) * float(size_t(eps_norm))));
        std::bernoulli_distribution dist(epsilon);

        for (size_t s=0; s<params.episode_length; s++) {
            Tensor input = environment->get_observation_space();
            input += 0.0001;

            // Get value prediction (singleton tensor)
            auto value_predict = critic->forward(input);

            // Get action distribution of the policy (shape of action space)
            auto log_probabilities = actor->forward(input);
            auto probabilities = torch::exp(log_probabilities);

            int64_t choice;

            if (dist(generator) == 1) {
                choice = choice_dist(generator);
                cerr << "random" << '\n';
            }
            else {
                // choice = torch::argmax(probabilities).item<int64_t>();
                choice = torch::multinomial(probabilities, 1).item<int64_t>();
            }

            environment->step(choice);

            float reward = environment->get_reward();

            episode.update(log_probabilities, value_predict, choice, reward);

            if (environment->is_terminated() or environment->is_truncated()) {
                break;
            }
        }

        e++;

        auto td_loss = episode.compute_td_loss(params.gamma, false, true, environment->is_terminated());
        auto entropy_loss = episode.compute_entropy_loss(false, false);

        auto actor_loss = td_loss - params.lambda*entropy_loss;
        auto critic_loss = episode.compute_critic_loss(params.gamma, false, environment->is_terminated());

        // Print some stats, increment loss using episode, update model if batch_size accumulated
        cerr << std::left
        << std::setw(8)  << "episode" << std::setw(8) << e
        << std::setw(8)  << "length" << std::setw(6) << episode.get_size()
        << std::setw(14) << "entropy_loss" << std::setw(12) << entropy_loss.item<float>()*params.lambda
        << std::setw(14) << "avg_entropy" << std::setw(12) << entropy_loss.item<float>()/float(episode.get_size())
        << std::setw(8)  << "td_loss " << std::setw(12) << td_loss.item<float>()
        << std::setw(14) << "critic_loss" << std::setw(12) << critic_loss.item<float>()
        << std::setw(10) << "epsilon " << std::setw(10) << epsilon << '\n';

        actor_loss.backward();
        critic_loss.backward();

        // Periodically apply the accumulated gradient to the model
        if (e % params.batch_size == 0){
            optimizer_actor.step();
            optimizer_actor.zero_grad();
        }

        // Periodically apply the accumulated gradient to the model
        // if (e % std::max(params.batch_size/2,size_t(1)) == 0){
        if (e % params.batch_size == 0){
            optimizer_critic.step();
            optimizer_critic.zero_grad();
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

    while (e < params.n_episodes) {
        environment->reset();

        for (size_t s=0; s<params.episode_length; s++) {
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
