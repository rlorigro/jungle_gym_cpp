#pragma once

#include "Hyperparameters.hpp"
#include "LRScheduler.hpp"
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

    inline void set_lr(float lr);

public:
    inline PPOAgent(const Hyperparameters& hyperparams, shared_ptr<Model> actor, shared_ptr<Model> critic);
    inline void sample_trajectories(TensorEpisode& tensor_episode, shared_ptr<Environment> env, atomic<size_t>& n_steps, size_t max_steps);
    inline void train_cycle(vector<shared_ptr<Environment>>& envs, size_t n_steps);
    inline void cache_params();

    inline void train(shared_ptr<const Environment> env);
    inline void demo(shared_ptr<const Environment> env);
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


void PPOAgent::sample_trajectories(TensorEpisode& tensor_episode, shared_ptr<Environment> environment, atomic<size_t>& step_index, size_t n_steps){
    if (!environment) {
        throw std::runtime_error("ERROR: Environment pointer is null");
    }

    torch::NoGradGuard no_grad;

    Episode episode;
    bool terminated = false;
    bool truncated = false;

    int64_t n_truncated = 0;

    while (step_index.fetch_add(1) < n_steps + n_truncated) {
        Tensor input = environment->get_observation_space();

        // Get value prediction (singleton tensor)
        auto value_predict = torch::flatten(critic_old->forward(input));

        // Get action distribution of the policy (shape of action space)
        auto log_probabilities = torch::flatten(actor_old->forward(input));
        auto probabilities = torch::exp(log_probabilities);

        int64_t choice = torch::multinomial(probabilities, 1).item<int64_t>();

        environment->step(choice);

        float reward = environment->get_reward();
        terminated = environment->is_terminated();
        truncated = environment->is_truncated();

        // If truncated, this won't actually append the state/action/etc, just update truncated vector of values
        episode.update(input, log_probabilities, value_predict, choice, reward, terminated, truncated);

        // Since truncations don't append observations, we add some padding every time one is encountered (usually rare)
        n_truncated += int64_t(truncated);

        if (truncated or terminated) {
            environment->reset();
        }
    }

    if (not (truncated or terminated)) {
        // When sampling episodes, we pause for training, but training (GAE etc) needs a T+1 value estimate
        Tensor input = environment->get_observation_space();
        auto value_predict = torch::flatten(critic_old->forward(input));
        episode.update_truncated(value_predict);
    }

    episode.to_tensor(tensor_episode);
}


void PPOAgent::set_lr(float lr) {
    for (auto& group : optimizer_actor.param_groups()) {
        group.options().set_lr(lr);
    }
    for (auto& group : optimizer_critic.param_groups()) {
        group.options().set_lr(lr);
    }
}


void PPOAgent::train(shared_ptr<const Environment> env){
    size_t n_steps_total = hyperparams.n_steps;
    size_t cycle_length = hyperparams.n_steps_per_cycle;
    size_t n_cycles = n_steps_total/cycle_length;

    // Initialize environments in bulk, upfront because their true episode length may be longer than the sampling length
    vector<shared_ptr<Environment> > envs;

    for (size_t i=0; i<hyperparams.n_threads; i++) {
        envs.emplace_back(env->clone());
    }

    const double lr_0 = hyperparams.learn_rate;
    const double lr_n = hyperparams.learn_rate_final;

    cerr << "lr_0: " << lr_0 << '\n';
    cerr << "lr_n: " << lr_n << '\n';

    LinearLRScheduler scheduler(lr_0, lr_n, n_cycles);

    for (size_t i=0; i<n_cycles; i++) {
        cache_params();
        train_cycle(envs, cycle_length);

        cerr << 100.0*float(i+1)/float(n_cycles) << "%" << '\n';

        float lr = float(scheduler.next());
        set_lr(lr);

        cerr << "learn rate: " << lr << '\n';
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
void PPOAgent::train_cycle(vector<shared_ptr<Environment> >& envs, size_t n_steps){
    if (envs.size() < hyperparams.n_threads) {
        throw runtime_error("ERROR: provided envs " + std::to_string(envs.size()) + " insufficient for n_threads "  + std::to_string(hyperparams.n_threads));
    }

    // Temporarily set torch MP threads to 1, so that trajectory sampling can be vanilla multithreaded
    torch::set_num_threads(1);

    atomic<size_t> step_index = 0;

    vector<thread> threads;
    for (size_t i=0; i<hyperparams.n_threads; i++) {
        if (!envs[i]) {
            throw runtime_error("ERROR: Environment pointer is null");
        }

        episodes[i].clear();
        threads.emplace_back(&PPOAgent::sample_trajectories, this, std::ref(episodes[i]), envs[i], std::ref(step_index), n_steps);
    }

    for (auto& t: threads) {
        t.join();
    }

    // Return the torch MP thread count to hyperparams.n_threads so that batched inference and training has max throughput
    torch::set_num_threads(hyperparams.n_threads);

    TensorEpisode episode(episodes);

    auto n_episodes = episode.get_n_episodes();
    auto total_reward = episode.rewards.sum().item<float>();

    cerr << "avg episode reward: " << total_reward / float(n_episodes) << '\n';
    cerr << "avg episode length: " << float(episode.size) / float(n_episodes) << '\n';

    // Still needed for Critic
    episode.compute_td_rewards(hyperparams.gamma);

    for (size_t i=0; i<hyperparams.n_epochs; i++) {
        int64_t b = 0;

        episode.for_each_batch(hyperparams.batch_size, [&](TensorEpisode& batch){
            auto action_dists = actor->forward(batch.states);
            batch.value_predictions = critic->forward(batch.states).squeeze();

            auto clip_loss = batch.compute_clip_loss(action_dists, 0.95, hyperparams.gamma, 0.2, true);
            auto entropy_loss = batch.compute_entropy_loss(action_dists, true, true);

            auto actor_loss = clip_loss + hyperparams.lambda*entropy_loss;
            auto critic_loss = 0.5*batch.compute_critic_loss(true);

            // if (not hyperparams.silent) {
            if (not hyperparams.silent and b == 0) {
                auto reward_avg = torch::sum(batch.rewards).item<float>()/float(batch.size);

                // Print some stats, increment loss using episode, update model if batch_size accumulated
                cerr << std::setprecision(3) << std::left
                << std::setw(7) << i
                << std::setw(7) << b
                << std::setw(10) << "l_entropy" << std::setw(12) << entropy_loss.item<float>()*hyperparams.lambda
                << std::setw(8) << "entropy" << std::setw(12) << -entropy_loss.item<float>()
                << std::setw(7) << "l_clip " << std::setw(12) << clip_loss.item<float>()
                << std::setw(9) << "l_critic" << std::setw(12) << critic_loss.item<float>()
                << std::setw(14) << "reward_avg" << std::setw(10) << reward_avg << '\n';
            }

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


void PPOAgent::demo(shared_ptr<const Environment> env){
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