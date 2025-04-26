#pragma once

#include "Hyperparameters.hpp"
#include "Environment.hpp"
#include "Episode.hpp"
#include "Policy.hpp"

#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <thread>

using std::filesystem::path;
using std::runtime_error;
using std::make_shared;
using std::shared_ptr;
using std::function;
using std::thread;


namespace JungleGym{


/**
 * Object which handles the training and testing of the PPO RL algorithm
 */
class PPOAgent {
    shared_ptr<Model> actor_old;
    shared_ptr<Model> critic_old;

    shared_ptr<Model> actor;
    shared_ptr<Model> critic;

    torch::optim::AdamW optimizer_actor;
    torch::optim::AdamW optimizer_critic;

    vector<TensorEpisode> episodes;

    Hyperparameters hyperparams;

public:
    inline PPOAgent(const Hyperparameters& hyperparams, shared_ptr<Model> actor, shared_ptr<Model> critic);
    inline void sample_trajectories(TensorEpisode& tensor_episode, shared_ptr<const Environment> env, atomic<size_t>& n_steps, size_t max_steps);
    inline void train_cycle(shared_ptr<const Environment> env, size_t n_steps);
    inline void train(shared_ptr<const Environment> env);
    inline void test(shared_ptr<const Environment> env);
    inline void save(const path& output_path) const;
    inline void load(const path& actor_path, const path& critic_path);
};


void PPOAgent::save(const path& output_dir) const{
    if (not std::filesystem::exists(output_dir)) {
        std::filesystem::create_directories(output_dir);
    }

    path actor_path = output_dir / "actor.pt";
    path critic_path = output_dir / "critic.pt";

    torch::save(actor, actor_path);
    torch::save(critic, critic_path);
}


inline void PPOAgent::load(const path& actor_path, const path& critic_path) {
    torch::load(actor, actor_path);
    torch::load(critic, critic_path);
}


PPOAgent::PPOAgent(const Hyperparameters& hyperparams, shared_ptr<Model> actor, shared_ptr<Model> critic):
        actor(actor),
        critic(critic),
        optimizer_actor(actor->parameters()),
        optimizer_critic(critic->parameters()),
        episodes(hyperparams.n_threads),
        hyperparams(hyperparams)
{
    if (!actor) {
        throw runtime_error("ERROR: actor pointer is null");
    }
    if (!critic) {
        throw runtime_error("ERROR: critic pointer is null");
    }
}


void PPOAgent::sample_trajectories(TensorEpisode& tensor_episode, shared_ptr<const Environment> env, atomic<size_t>& step_index, size_t n_steps){
    if (!env) {
        throw std::runtime_error("ERROR: Environment pointer is null");
    }

    // make a copy of the environment
    shared_ptr<Environment> environment = env->clone();
    environment->reset();

    Episode episode;

    while (step_index.fetch_add(1) < n_steps) {
        Tensor input = environment->get_observation_space();
        input += 0.0001;

        // Get value prediction (singleton tensor)
        auto value_predict = torch::flatten(critic->forward(input));

        // Get action distribution of the policy (shape of action space)
        auto log_probabilities = torch::flatten(actor->forward(input));
        auto probabilities = torch::exp(log_probabilities);

        int64_t choice = torch::multinomial(probabilities, 1).item<int64_t>();

        environment->step(choice);

        float reward = environment->get_reward();
        bool done = environment->is_terminated() or environment->is_truncated();

        cerr << step_index << ',' << reward << ',' << done << ',' << environment->is_terminated() << ',' << environment->is_truncated() << '\n';

        episode.update(input, log_probabilities, value_predict, choice, reward, done);

        if (done) {
            environment->reset();
        }
    }

    episode.to_tensor(tensor_episode);
}


void PPOAgent::train(shared_ptr<const Environment> env){
    train_cycle(env, hyperparams.n_threads*16);
}


/**
* This performs all the sequential steps involved for one "cycle" of training:
* - Set torch threads to 1
* - Copy params from theta into theta_old
* - Launch n threads to generate trajectories (inference only, using theta_old)
* - Set torch threads to n_threads
* - Aggregate trajectories into contiguous batched Tensors and compute TD reward
* - Perform batched training over the trajectories for K epochs
*/
void PPOAgent::train_cycle(shared_ptr<const Environment> env, size_t n_steps){
    if (!env) {
        throw runtime_error("ERROR: Environment pointer is null");
    }

    // Temporarily set torch MP threads to 1, so that trajectory sampling can be vanilla multithreaded
    int n_torch_threads = torch::get_num_threads();
    torch::set_num_threads(1);

    if (not hyperparams.silent) {
        cerr << "n_torch_threads: " << n_torch_threads << " ... setting to 1 for PPO trajectory generation." << '\n';
    }

    atomic<size_t> step_index = 0;

    if (n_steps % hyperparams.n_threads != 0 or n_steps < hyperparams.n_threads) {
        throw runtime_error("ERROR: n_steps must be a multiple of n_threads, and n_steps must not be less than n_threads");
    }

    vector<thread> threads;
    for (size_t i=0; i<hyperparams.n_threads; i++) {
        episodes[i].clear();
        threads.emplace_back(&PPOAgent::sample_trajectories, this, std::ref(episodes[i]), env, std::ref(step_index), n_steps);
    }

    for (auto& t: threads) {
        t.join();
    }

    if (not hyperparams.silent) {
        cerr << "resetting torch threads to: " << n_torch_threads << '\n';
    }

    // Return the torch MP thread count to default so that batched inference and training has max throughput
    torch::set_num_threads(n_torch_threads);

    TensorEpisode episode(episodes);

    cerr << "log_action_distributions: " << episode.log_action_distributions.sizes() << '\n';
    cerr << "states: " << episode.states.sizes() << '\n';
    cerr << "value_predictions: " << episode.value_predictions.sizes() << '\n';
    cerr << "actions: " << episode.actions.sizes() << '\n';
    cerr << "rewards: " << episode.rewards.sizes() << '\n';
    cerr << "td_rewards: " << episode.td_rewards.sizes() << '\n';
    cerr << "mask: " << episode.mask.sizes() << '\n';

    episode.compute_td_rewards(hyperparams.gamma);
    cerr << "mask: \n" << episode.mask << '\n';
    cerr << "rewards: \n" << episode.rewards << '\n';
    cerr << "td_rewards: \n" << episode.td_rewards << '\n';

    episode.for_each_batch(4, [&](TensorEpisode& batch){
        cerr << "states: " << batch.states.sizes() << '\n';
        cerr << "td_rewards: " << batch.td_rewards.sizes() << '\n';
        cerr << "mask: " << batch.mask.sizes() << '\n';
        cerr << "mask:\n" << batch.mask << '\n';

        auto action_dists = actor->forward(batch.states);
        batch.value_predictions = critic->forward(batch.states);

        cerr << "action_dists: " << action_dists.sizes() << '\n';
        cerr << "values: " << batch.value_predictions.sizes() << '\n';

        auto critic_loss = batch.compute_critic_loss(true);
        auto actor_loss = batch.compute_clip_loss(action_dists, 0.2, true);
        auto entropy = batch.compute_entropy_loss(action_dists, true, true);
    });


}


void PPOAgent::test(shared_ptr<const Environment> env){
    throw runtime_error("PPOAgent::test NOT IMPLEMENTED");

    if (!env) {
        throw std::runtime_error("ERROR: Environment pointer is null");
    }

    actor->eval();
    critic->eval();

    shared_ptr<Environment> environment = env->clone();

    std::thread t(std::bind(&Environment::render, environment, false));

    while (true) {
        environment->reset();

        while (true) {
            auto input = environment->get_observation_space();
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