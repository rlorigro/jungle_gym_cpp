#pragma once

#include <iostream>
#include "Hyperparameters.hpp"
#include "Environment.hpp"
#include "Episode.hpp"
#include "Policy.hpp"
#include "memory"
#include <filesystem>

using std::filesystem::path;
using std::shared_ptr;
using std::make_shared;


namespace JungleGym{


/**
 * Object which handles the training and testing of the PG (policy gradient) RL model/algorithm
 */
class PGAgent {
    shared_ptr<Model> actor;
    Episode episode;
    torch::optim::RMSprop optimizer;
    Hyperparameters hyperparams;

public:
    inline PGAgent(const Hyperparameters& hyperparams, shared_ptr<Model> actor);
    inline void train(shared_ptr<const Environment> env);
    inline void test(shared_ptr<const Environment> env);
    inline void save(const path& output_path) const;
    inline void load(const path& actor_path);

};


void PGAgent::save(const path& output_dir) const{
    if (not std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }

    path actor_path = output_dir / "actor.pt";

    torch::save(actor, actor_path);
}


inline void PGAgent::load(const path& actor_path) {
    torch::load(actor, actor_path);
}


PGAgent::PGAgent(const Hyperparameters& hyperparams, shared_ptr<Model> actor):
        actor(actor),
        episode(),
        optimizer(actor->parameters(), torch::optim::RMSpropOptions(hyperparams.learn_rate)),
        hyperparams(hyperparams)
{
    if (!actor) {
        throw std::runtime_error("ERROR: actor pointer is null");
    }
}


void PGAgent::train(shared_ptr<const Environment> env){
    if (!env) {
        throw std::runtime_error("ERROR: Environment pointer is null");
    }

    shared_ptr<Environment> environment = env->clone();

    // Epsilon is adjusted on a schedule, not fixed
    // TODO: un-remove epsilon greedy implementation? make optional?
    float epsilon = 0;

    // Used for deciding if we want to act greedily or sample randomly
    mt19937 generator(1337);
    std::uniform_int_distribution<int64_t> choice_dist(0,4-1);

    size_t e = 0;

    float eps_terminal = 0.01;

    // log_b(x) = log_e(x)/log_e(b)
    float eps_norm = log(0.01)/log(0.99);

    if (not hyperparams.silent) {
        cerr << "Using eps_norm: " << eps_norm << " for eps_terminal: " << eps_terminal << '\n';
    }

    while (e < hyperparams.n_episodes) {
        environment->reset();
        episode.clear();

        // exponential decay that terminates at ~0.01
        // epsilon = pow(0.99,(float(e)/float(hyperparams.n_episodes)) * float(size_t(eps_norm))));
        std::bernoulli_distribution dist(epsilon);

        for (size_t s=0; s<hyperparams.episode_length; s++) {

            Tensor input = environment->get_observation_space();
            input += 0.0001;

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

            // cerr << "probabilities: \n" << probabilities << '\n';
            // cerr << "choice: " << choice << "\n";
            // cerr << "prev_action: " << environment->get_prev_action() << "\n";
            // cerr << "length: " << episode.get_size() << "\n";
            // cerr << "epsilon: " << epsilon << "\n";

            environment->step(choice);

            float reward = environment->get_reward();

            episode.update(log_probabilities, choice, reward);

            if (environment->is_terminated() or environment->is_truncated()) {
                break;
            }
        }

        e++;

        auto td_loss = episode.compute_td_loss(hyperparams.gamma, false, false);
        auto entropy_loss = episode.compute_entropy_loss(false, false);

        if (not hyperparams.silent) {
            // Print some stats, increment loss using episode, update model if batch_size accumulated
            cerr << std::left
            << std::setw(8) << "episode" << std::setw(8) << e
            << std::setw(8) << "length" << std::setw(6) << episode.get_size()
            << std::setw(14) << "entropy_loss" << std::setw(12) << entropy_loss.item<float>()*hyperparams.lambda
            << std::setw(8) << "td_loss " << std::setw(12) << td_loss.item<float>()
            << std::setw(10) << "epsilon " << std::setw(10) << epsilon << '\n';
        }

        auto loss = td_loss + hyperparams.lambda*entropy_loss;
        loss.backward();

        // Occasionally apply the accumulated gradient to the model
        if (e % hyperparams.batch_size == 0){
            optimizer.step();
            optimizer.zero_grad();
        }
    }
}


void PGAgent::test(shared_ptr<const Environment> env){
    if (!env) {
        throw std::runtime_error("ERROR: Environment pointer is null");
    }
    torch::NoGradGuard no_grad;

    actor->eval();

    shared_ptr<Environment> environment = env->clone();

    auto prev_action = environment->get_action_space();

    size_t e = 0;
    std::thread t(std::bind(&Environment::render, environment, false));

    while (e < hyperparams.n_episodes) {
        environment->reset();

        for (size_t s=0; s<hyperparams.episode_length; s++) {
            auto input = environment->get_observation_space();
            input += 0.0001;

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
