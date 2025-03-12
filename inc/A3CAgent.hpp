#pragma once

#include <iostream>
#include "Hyperparameters.hpp"
#include "A2CAgent.hpp"
#include "Environment.hpp"
#include "Episode.hpp"
#include "Policy.hpp"
#include "memory"

using std::shared_ptr;
using std::make_shared;
using std::shared_lock;
using std::unique_lock;
using std::thread;


namespace JungleGym{


/**
 * Object which handles the training and testing of the A2C (policy gradient) RL model/algorithm
 */
class A3CAgent {
	vector <shared_ptr <Model> > thread_actors;
	vector <shared_ptr <Model> > thread_critics;

    shared_ptr<Model> main_actor;
    shared_ptr<Model> main_critic;

    vector<torch::optim::RMSprop> thread_optimizers_actor;
    vector<torch::optim::RMSprop> thread_optimizers_critic;

    torch::optim::RMSprop optimizer_actor;
    torch::optim::RMSprop optimizer_critic;

    const Hyperparameters params;
    shared_mutex m;
    atomic<size_t> episode_index;
	size_t n_threads;

public:
    inline A3CAgent(const Hyperparameters& params, shared_ptr<Model> actor, shared_ptr<Model> critic, size_t n_threads);
    inline void train_thread(shared_ptr<const Environment> env, size_t thread_id);
    inline void train(shared_ptr<const Environment> env);
    inline void test(shared_ptr<const Environment> env);
};


A3CAgent::A3CAgent(const Hyperparameters& params, shared_ptr<Model> actor, shared_ptr<Model> critic, size_t n_threads):
        main_actor(actor),
        main_critic(critic),
        optimizer_actor(actor->parameters(), torch::optim::RMSpropOptions(params.learn_rate)),
        optimizer_critic(critic->parameters(), torch::optim::RMSpropOptions(params.learn_rate)),
        params(params),
        n_threads(n_threads)
{
    if (!actor) {
        throw std::runtime_error("ERROR: actor pointer is null");
    }
    if (!critic) {
        throw std::runtime_error("ERROR: critic pointer is null");
    }

	// Do this if decided that separate optimizers are needed
    for (size_t i = 0; i < n_threads; i++) {
    	thread_actors.emplace_back(actor->clone());
		thread_critics.emplace_back(critic->clone());
        thread_optimizers_actor.emplace_back(actor->parameters(), torch::optim::RMSpropOptions(params.learn_rate));
        thread_optimizers_critic.emplace_back(critic->parameters(), torch::optim::RMSpropOptions(params.learn_rate));
    }
}


void A3CAgent::train(shared_ptr<const Environment> env){
	atomic<size_t> batch_index;
    vector<thread> threads;

    for (size_t i=0; i<n_threads; i++){
    	threads.emplace_back(std::bind(&A3CAgent::train_thread, this, env, i));
    }

    for (auto& t : threads){
    	t.join();
    }
}


void A3CAgent::train_thread(shared_ptr<const Environment> env, size_t thread_id){
    if (!env) {
        throw std::runtime_error("ERROR: Environment pointer is null");
    }

    auto& critic = thread_critics.at(thread_id);
	auto& actor = thread_actors.at(thread_id);

    // TODO: decide if common storage is actually needed as above ^ or just have threads clone upon creation

    // TODO: implement gradient copying function and step() within locked context

    Episode episode;

    // make a copy of the environment
    shared_ptr<Environment> environment = env->clone();
    environment->reset();

    // Used for deciding if we want to act greedily or sample randomly
    mt19937 generator(1337);
    std::uniform_int_distribution<int64_t> choice_dist(0,4-1);

    size_t e = episode_index.fetch_add(1);

    while (e < params.n_episodes) {
        episode.clear();

        for (size_t s=0; s<params.episode_length; s++) {
            Tensor input = environment->get_observation_space();
            input += 0.0001;

            shared_lock lock(m);
            // Get value prediction (singleton tensor)
            auto value_predict = critic->forward(input);

            // Get action distribution of the policy (shape of action space)
            auto log_probabilities = actor->forward(input);
            lock.unlock();

            auto probabilities = torch::exp(log_probabilities);

            int64_t choice;

            // choice = torch::argmax(probabilities).item<int64_t>();
            choice = torch::multinomial(probabilities, 1).item<int64_t>();

            environment->step(choice);

            float reward = environment->get_reward();

            episode.update(log_probabilities, value_predict, choice, reward);

            if (environment->is_terminated() or environment->is_truncated()) {
                environment->reset();
                break;
            }
        }

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
        << std::setw(14) << "critic_loss" << std::setw(12) << critic_loss.item<float>() << '\n';

        unique_lock lock(m);
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

        e = episode_index.fetch_add(1);
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
            auto value_predict = main_critic->forward(input);

            // Get action distribution of the policy (shape of action space)
            auto log_probabilities = main_actor->forward(input);
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
