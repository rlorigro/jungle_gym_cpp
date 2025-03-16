#include "cpptrace/from_current.hpp"
#include "Hyperparameters.hpp"
#include "Episode.hpp"
#include "SnakeEnv.hpp"
#include "Policy.hpp"
#include "A2CAgent.hpp"
#include "A3CAgent.hpp"
#include "PGAgent.hpp"
#include "CLI11.hpp"

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

#include <iostream>
#include <stdexcept>

using std::runtime_error;
using std::cerr;
using std::min;


void train_and_test(Hyperparameters& hyperparams, string type){
    // For now we fix the grid size
    size_t w = 10;

    if (type == "pg") {
        shared_ptr<SnakeEnv> env = make_shared<SnakeEnv>(w,w);
        shared_ptr<SimpleConv> actor = make_shared<SimpleConv>(w,w,4,4);

        PGAgent agent(hyperparams, actor);
        agent.train(env);
        agent.test(env);
    }
    else if (type == "a2c") {
        shared_ptr<SnakeEnv> env = make_shared<SnakeEnv>(w,w);
        shared_ptr<SimpleConv> actor = make_shared<SimpleConv>(w,w,4,4);
        shared_ptr<SimpleConv> critic = make_shared<SimpleConv>(w,w,4,1);

        A2CAgent agent(hyperparams, actor, critic);
        agent.train(env);
        agent.test(env);
    }
    else if (type == "a3c") {
        shared_ptr<SnakeEnv> env = make_shared<SnakeEnv>(w,w);
        shared_ptr<SimpleConv> actor = make_shared<SimpleConv>(w,w,4,4);
        shared_ptr<SimpleConv> critic = make_shared<SimpleConv>(w,w,4,1);

        A3CAgent agent(hyperparams, actor, critic);
        agent.train(env);
        agent.test(env);
    }
    else {
        throw runtime_error("ERROR: Unsupported hyperparameter type: " + type);
    }
}


int main(int argc, char* argv[]){
    CLI::App app{"App description"};
    Hyperparameters params;
    string type;

    app.add_option(
            "--type",
            type,
            "What RL algorithm to use:\n"
            "\t'pg' = policy gradient\n"
            "\t'ac' = actor critic");

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

    app.add_option(
            "--lambda",
            params.lambda,
            "lambda");

    try{
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    CPPTRACE_TRY {
        train_and_test(params, type);
    } CPPTRACE_CATCH(const std::exception& e) {
        std::cerr<<"Exception: "<<e.what()<<std::endl;
        cpptrace::from_current_exception().print_with_snippets();
    }

    return 0;
}
