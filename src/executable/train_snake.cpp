#include "cpptrace/from_current.hpp"
#include "Hyperparameters.hpp"
#include "Episode.hpp"
#include "SnakeEnv.hpp"
#include "Policy.hpp"
#include "A2CAgent.hpp"
#include "A3CAgent.hpp"
#include "PPOAgent.hpp"
#include "PGAgent.hpp"
#include "CLI11.hpp"
#include "misc.hpp"

// This must be included to properly implement mimalloc
#include "mimalloc_conditional_override.hpp"

#include <ranges>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <thread>
#include <random>
#include <deque>
#include <vector>
#include <array>
#include <string>

using std::this_thread::sleep_for;
using std::runtime_error;
using std::random_device;
using std::mt19937;
using std::deque;
using std::vector;
using std::pair;
using std::cerr;
using std::array;
using std::mutex;
using std::string;

using namespace JungleGym;

using std::runtime_error;
using std::cerr;
using std::min;


void train_and_test(Hyperparameters& hyperparams, string type, bool skip_test){
    hyperparams.silent = false;

    // For now we fix the grid size
    size_t w = 10;
    path output_dir = std::filesystem::weakly_canonical("output") / get_timestamp();

    if (type == "pg") {
        shared_ptr<SnakeEnv> env = make_shared<SnakeEnv>(w,w);

        int64_t out_size = env->get_action_space().numel();
        cerr << "action space: " << out_size << '\n';
        shared_ptr<SimpleConv> actor = make_shared<SimpleConv>(w,w,4,out_size);

        PGAgent agent(hyperparams, actor);
        agent.train(env);
        agent.save(output_dir);

        if (not skip_test){
            agent.test(env);
        }
    }
    else if (type == "a2c") {
        shared_ptr<SnakeEnv> env = make_shared<SnakeEnv>(w,w);

        int64_t out_size = env->get_action_space().numel();
        cerr << "action space: " << out_size << '\n';
        shared_ptr<SimpleConv> actor = make_shared<SimpleConv>(w,w,4,out_size);
        shared_ptr<SimpleConv> critic = make_shared<SimpleConv>(w,w,4,1);

        A2CAgent agent(hyperparams, actor, critic);
        agent.train(env);
        agent.save(output_dir);

        if (not skip_test){
            agent.test(env);
        }
    }
    else if (type == "a3c") {
        shared_ptr<SnakeEnv> env = make_shared<SnakeEnv>(w,w);

        int64_t out_size = env->get_action_space().numel();
        cerr << "action space: " << out_size << '\n';
        shared_ptr<SimpleConv> actor = make_shared<SimpleConv>(w,w,4,out_size);
        shared_ptr<SimpleConv> critic = make_shared<SimpleConv>(w,w,4,1);

        A3CAgent agent(hyperparams, actor, critic);
        agent.train(env);
        agent.save(output_dir);

        if (not skip_test){
            agent.test(env);
        }
    }
    else if (type == "ppo") {
        shared_ptr<SnakeEnv> env = make_shared<SnakeEnv>(w,w);

        int64_t out_size = env->get_action_space().numel();
        cerr << "action space: " << out_size << '\n';
        shared_ptr<SimpleConv> actor = make_shared<SimpleConv>(w,w,4,out_size);
        shared_ptr<SimpleConv> critic = make_shared<SimpleConv>(w,w,4,1);

        PPOAgent agent(hyperparams, actor, critic);
        agent.train(env);
        agent.save(output_dir);

        if (not skip_test){
            agent.test(env);
        }
    }
    else {
        throw runtime_error("ERROR: Unsupported hyperparameter type: " + type);
    }
}


int main(int argc, char* argv[]){
    CLI::App app{"App description"};
    Hyperparameters params;
    string type;
    bool skip_test = false;

    app.add_option(
            "--type",
            type,
            "What RL algorithm to use:\n"
            "\t'pg' = policy gradient\n"
            "\t'a3c' = actor critic"
            "\t'ppo' = proximal policy optimization (using L_CLIP)"
            );

    app.add_option(
            "--episode_length",
            params.episode_length,
            "episode_length");

    app.add_option(
            "--n_episodes",
            params.n_episodes,
            "n_episodes");

    app.add_option(
            "--n_threads",
            params.n_threads,
            "Number of threads to use. Only applies to asynchronous algos such as A3C");

    app.add_option(
            "--batch_size",
            params.batch_size,
            "batch_size");

    app.add_option(
            "--gamma",
            params.gamma,
            "gamma");

    app.add_option(
            "--learn_rate",
            params.learn_rate,
            "learn_rate");

    auto* lr_final = app.add_option(
            "--learn_rate_final",
            params.learn_rate_final,
            "learn_rate at end of training, scheduled linearly");

    app.add_option(
            "--lambda",
            params.lambda,
            "lambda coeff for entropy loss");

    app.add_option(
            "--n_steps",
            params.n_steps,
            "n_steps total in training, ONLY FOR PPO");

    app.add_option(
            "--n_steps_per_cycle",
            params.n_steps_per_cycle,
            "n_steps_per_cycle, ONLY FOR PPO");

    app.add_option(
            "--n_epochs",
            params.n_epochs,
            "n_epochs, ONLY FOR PPO");

    app.add_flag(
            "--skip_test",
            skip_test,
            "Dont run the test method (which typically renders the environment)");

    try{
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    if (lr_final->count() == 0) {
        params.learn_rate_final = params.learn_rate;
        cerr << "learn_rate_final set to learn_rate: " << params.learn_rate_final << '\n';
    }

    CPPTRACE_TRY {
        train_and_test(params, type, skip_test);
    } CPPTRACE_CATCH(const std::exception& e) {
        std::cerr<<"Exception: "<<e.what()<<std::endl;
        cpptrace::from_current_exception().print_with_snippets();
    }

    return 0;
}
