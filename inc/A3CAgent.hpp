#pragma once

#include "Hyperparameters.hpp"
#include "RMSPropAsync.hpp"
#include "Environment.hpp"
#include "A2CAgent.hpp"
#include "Episode.hpp"
#include "Policy.hpp"

#include <filesystem>
#include <iostream>
#include <memory>

using std::filesystem::path;
using std::runtime_error;
using std::make_shared;
using std::shared_ptr;
using std::thread;


namespace JungleGym{


/**
 * Object which handles the training and testing of the A3C (policy gradient) RL model/algorithm
 */
class A3CAgent {
    shared_ptr<Model> actor;
    shared_ptr<Model> critic;

    RMSPropAsync optimizer_actor;
    RMSPropAsync optimizer_critic;

    Hyperparameters params;

public:
    inline A3CAgent(const Hyperparameters& params, shared_ptr<Model> actor, shared_ptr<Model> critic);
    inline void train(shared_ptr<const Environment> env);
    inline void test(shared_ptr<const Environment> env);
    inline void save(const path& output_path) const;
    inline void load(const path& actor_path, const path& critic_path);
};


void A3CAgent::save(const path& output_dir) const{
    if (not std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }

    path actor_path = output_dir / "actor.pt";
    path critic_path = output_dir / "critic.pt";

    torch::save(actor, actor_path);
    torch::save(critic, critic_path);
}


inline void A3CAgent::load(const path& actor_path, const path& critic_path) {
    torch::load(actor, actor_path);
    torch::load(critic, critic_path);
}


A3CAgent::A3CAgent(const Hyperparameters& params, shared_ptr<Model> actor, shared_ptr<Model> critic):
        actor(actor),
        critic(critic),
        optimizer_actor(actor->parameters(), params.learn_rate),
        optimizer_critic(critic->parameters(), params.learn_rate),
        params(params)
{
    if (!actor) {
        throw runtime_error("ERROR: actor pointer is null");
    }
    if (!critic) {
        throw runtime_error("ERROR: critic pointer is null");
    }
}


void A3CAgent::train(shared_ptr<const Environment> env){
    if (!env) {
        throw runtime_error("ERROR: Environment pointer is null");
    }

    // Returns the maximum number of threads that may be used in a parallel region
    int n_torch_threads = torch::get_num_threads();

    // Sets the number of threads to be used in parallel region
    torch::set_num_threads(1);

    if (not params.silent) {
        cerr << "n_torch_threads: " << n_torch_threads << " ... setting to 1 for A3C training." << '\n';
    }

    // Make a copy of the environment
    shared_ptr<Environment> environment = env->clone();
    environment->reset();
    atomic<size_t> episode;

    auto sync_fn = [&](shared_ptr<Model> actor_worker, shared_ptr<Model> critic_worker, size_t& e) {
        e = episode.fetch_add(1);

        // The RMSPropAsync optimizer handles the locking of the parameters during global/shared param updates
        optimizer_critic.step(critic_worker->parameters());
        optimizer_actor.step(actor_worker->parameters());

        critic_worker->zero_grad();
        actor_worker->zero_grad();

        // Sync the worker with the shared params
        optimizer_critic.get_params(critic_worker);
        optimizer_actor.get_params(actor_worker);

        // This should be true to block the worker from stepping its own local optimizer and episode counter
        return true;
    };

    vector<thread> threads;
    for (size_t i=0; i<params.n_threads; i++) {
        threads.emplace_back([&]() {
            A2CAgent worker(params, actor->clone(), critic->clone());
            worker.train(environment, sync_fn);
        });
    }

    for (auto& t: threads) {
        t.join();
    }

    if (not params.silent) {
        cerr << "resetting torch threads to: " << n_torch_threads << '\n';
    }

    torch::set_num_threads(n_torch_threads);
}


void A3CAgent::test(shared_ptr<const Environment> env){
    if (!env) {
        throw std::runtime_error("ERROR: Environment pointer is null");
    }

    actor->eval();
    critic->eval();

    shared_ptr<Environment> environment = env->clone();

    auto prev_action = environment->get_action_space();

    std::thread t(std::bind(&Environment::render, environment, false));

    while (true) {
        environment->reset();

        while (true) {
            auto input = environment->get_observation_space().clone();
            input += 0.0001;

            // Get value prediction (singleton tensor)
            auto value_predict = critic->forward(input);

            // Get action distribution of the policy (shape of action space)
            auto log_probabilities = actor->forward(input);
            auto probabilities = torch::exp(log_probabilities);

            cerr << probabilities << '\n';

            int64_t choice = torch::argmax(probabilities).item<int64_t>();
            // int64_t choice = torch::multinomial(probabilities, 1).item<int64_t>();

            environment->step(choice);

            if (environment->is_terminated() or environment->is_truncated()) {
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
    }

    t.join();
}


}
