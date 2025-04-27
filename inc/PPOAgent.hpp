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
    inline void cache_params();
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
        optimizer_actor(actor->parameters(), torch::optim::AdamWOptions(hyperparams.learn_rate).weight_decay(0.001)),
        optimizer_critic(critic->parameters(), torch::optim::AdamWOptions(hyperparams.learn_rate).weight_decay(0.001)),
        episodes(hyperparams.n_threads),
        hyperparams(hyperparams)
{
    if (!actor) {
        throw runtime_error("ERROR: actor pointer is null");
    }
    if (!critic) {
        throw runtime_error("ERROR: critic pointer is null");
    }

    actor_old = actor->clone();
    critic_old = critic->clone();
}


void PPOAgent::cache_params() {
    torch::NoGradGuard no_grad;

    auto actor_old_params = actor_old->parameters();
    auto actor_params = actor->parameters();
    auto critic_old_params = critic_old->parameters();
    auto critic_params = critic->parameters();

    for (size_t i=0; i<actor_old_params.size(); i++) {
        auto& p_old = actor_old_params[i];
        auto& p_new = actor_params[i];
        p_old.copy_(p_new);
    }

    for (size_t i=0; i<critic_old_params.size(); i++) {
        auto& p_old = critic_old_params[i];
        auto& p_new = critic_params[i];
        p_old.copy_(p_new);
    }

    actor_old->eval();
    critic_old->eval();
}


void PPOAgent::sample_trajectories(TensorEpisode& tensor_episode, shared_ptr<const Environment> env, atomic<size_t>& step_index, size_t n_steps){
    if (!env) {
        throw std::runtime_error("ERROR: Environment pointer is null");
    }

    torch::NoGradGuard no_grad;

    cache_params();

    // make a copy of the environment
    shared_ptr<Environment> environment = env->clone();
    environment->reset();

    Episode episode;

    while (step_index.fetch_add(1) < n_steps) {
        Tensor input = environment->get_observation_space();
        input += 0.0001;

        // Get value prediction (singleton tensor)
        auto value_predict = torch::flatten(critic_old->forward(input));

        // Get action distribution of the policy (shape of action space)
        auto log_probabilities = torch::flatten(actor_old->forward(input));
        auto probabilities = torch::exp(log_probabilities);

        int64_t choice = torch::multinomial(probabilities, 1).item<int64_t>();

        environment->step(choice);

        float reward = environment->get_reward();
        bool done = environment->is_terminated() or environment->is_truncated();

        // if (not hyperparams.silent and step_index % 50 == 0) {
        //     cerr << std::setprecision(3) << std::left
        //     << std::setw(8) << step_index
        //     << std::setw(8) << reward
        //     << std::setw(8) << done << '\n';
        // }

        episode.update(input, log_probabilities, value_predict, choice, reward, done);

        if (done) {
            environment->reset();
        }
    }

    episode.to_tensor(tensor_episode);
}


void PPOAgent::train(shared_ptr<const Environment> env){
    size_t n_steps_total = 1'000'000;
    size_t cycle_length = 2048;
    size_t n_cycles = n_steps_total/cycle_length;

    for (size_t i=0; i<n_cycles; i++) {
        train_cycle(env, cycle_length);
        cerr << 100.0*float(i+1)/float(n_cycles) << "%" << '\n';
    }
}


/**
* This performs all the sequential steps involved for one "cycle" of training:
* - Set torch threads to 1
* - Copy params from theta into theta_old
* - Launch n threads to generate trajectories (inference only, using theta_old)
* - Set torch threads to n_threads
* - Aggregate trajectories into contiguous Tensors and compute TD reward
* - Perform batched training over the trajectories for K epochs
*/
void PPOAgent::train_cycle(shared_ptr<const Environment> env, size_t n_steps){
    if (!env) {
        throw runtime_error("ERROR: Environment pointer is null");
    }

    // Temporarily set torch MP threads to 1, so that trajectory sampling can be vanilla multithreaded
    torch::set_num_threads(1);

    atomic<size_t> step_index = 0;

    vector<thread> threads;
    for (size_t i=0; i<hyperparams.n_threads; i++) {
        episodes[i].clear();
        threads.emplace_back(&PPOAgent::sample_trajectories, this, std::ref(episodes[i]), env, std::ref(step_index), n_steps);
    }

    for (auto& t: threads) {
        t.join();
    }

    // Return the torch MP thread count to hyperparams.n_threads so that batched inference and training has max throughput
    torch::set_num_threads(hyperparams.n_threads);

    TensorEpisode episode(episodes);

    auto n_terminations = torch::sum(episode.mask).item<int64_t>();
    auto total_reward = torch::sum(episode.rewards).item<float>();

    cerr << "avg episode reward: " << total_reward / float(n_terminations) << '\n';
    cerr << "avg episode length: " << float(episode.size) / float(n_terminations) << '\n';

    // cerr << "log_action_distributions: " << episode.log_action_distributions.sizes() << '\n';
    // cerr << "states: " << episode.states.sizes() << '\n';
    // cerr << "value_predictions: " << episode.value_predictions.sizes() << '\n';
    // cerr << "actions: " << episode.actions.sizes() << '\n';
    // cerr << "rewards: " << episode.rewards.sizes() << '\n';
    // cerr << "td_rewards: " << episode.td_rewards.sizes() << '\n';
    // cerr << "mask: " << episode.mask.sizes() << '\n';

    episode.compute_td_rewards(hyperparams.gamma);

    for (size_t i=0; i<3; i++) {
        int64_t b = 0;

        episode.for_each_batch(hyperparams.batch_size, [&](TensorEpisode& batch){
            auto action_dists = actor->forward(batch.states);
            batch.value_predictions = critic->forward(batch.states);

            auto critic_loss = batch.compute_critic_loss(false);
            auto clip_loss = batch.compute_clip_loss(action_dists, 0.2, false);
            auto entropy_loss = batch.compute_entropy_loss(action_dists, false, true);

            // if (not hyperparams.silent) {
            if (not hyperparams.silent and b == 0) {
                auto reward_avg = torch::sum(batch.rewards).item<float>()/float(batch.size);

                // Print some stats, increment loss using episode, update model if batch_size accumulated
                cerr << std::setprecision(3) << std::left
                << std::setw(7) << i
                << std::setw(7) << b
                << std::setw(10) << "l_entropy" << std::setw(12) << entropy_loss.item<float>()*hyperparams.lambda/float(batch.size)
                << std::setw(8) << "entropy" << std::setw(12) << -entropy_loss.item<float>()/float(batch.size)
                << std::setw(7) << "l_clip " << std::setw(12) << clip_loss.item<float>()/float(batch.size)
                << std::setw(9) << "l_critic" << std::setw(12) << critic_loss.item<float>()/float(batch.size)
                << std::setw(14) << "reward_avg" << std::setw(10) << reward_avg << '\n';
            }

            auto actor_loss = clip_loss + hyperparams.lambda*entropy_loss;

            critic_loss.backward();
            actor_loss.backward();

            optimizer_critic.step();
            optimizer_critic.zero_grad();

            optimizer_actor.step();
            optimizer_actor.zero_grad();

            b++;
        });
    }


}


void PPOAgent::test(shared_ptr<const Environment> env){
    // throw runtime_error("PPOAgent::test NOT IMPLEMENTED");

    if (!env) {
        throw std::runtime_error("ERROR: Environment pointer is null");
    }

    torch::NoGradGuard no_grad;

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