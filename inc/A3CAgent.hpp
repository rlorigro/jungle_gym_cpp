#pragma once

#include <iostream>
#include "Hyperparameters.hpp"
#include "RMSPropAsync.hpp"
#include "Environment.hpp"
#include "Episode.hpp"
#include "Policy.hpp"
#include <memory>

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
};


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

    // Make a copy of the environment
    shared_ptr<Environment> environment = env->clone();
    environment->reset();
    atomic<size_t> episode;

    auto sync_fn = [&](shared_ptr<Model> actor_worker, shared_ptr<Model> critic_worker, size_t& e) {
        // The RMSPropAsync optimizer handles the locking of the parameters during global/shared param updates
        optimizer_critic.step(critic_worker->parameters());
        optimizer_actor.step(actor_worker->parameters());

        critic_worker->zero_grad();
        actor_worker->zero_grad();

        // Sync the worker with the shared params
        optimizer_critic.get_params(critic_worker);
        optimizer_actor.get_params(actor_worker);

        e = episode.fetch_add(1);

        // This should be true to block the worker from stepping its own local optimizer
        return true;
    };

    vector<thread> threads;
    for (size_t i=0; i<params.n_threads; i++) {
        threads.emplace_back([&]() {
            auto e = environment->clone();
            A2CAgent worker(params, actor->clone(), critic->clone());

            worker.train(environment, sync_fn);
        });
    }

    for (auto& t: threads) {
        t.join();
    }

}


void A3CAgent::test(shared_ptr<const Environment> env){
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
