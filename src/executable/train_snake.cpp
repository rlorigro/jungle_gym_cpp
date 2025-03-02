#include "cpptrace/from_current.hpp"
#include "ReplayBuffer.hpp"
#include "SnakeEnv.hpp"
#include "Policy.hpp"
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


class Hyperparameters {
public:
    string type = "pg";
    size_t episode_length = 200;
    size_t n_episodes = 2000;
    size_t batch_size = 16;
    float learn_rate = 5e-5;

    // For decay rate of TD recurrence
    float gamma = 0.1;

    // For weight of entropy term in the loss function
    float lambda = 0.01;
};


void train_actor_critic(std::shared_ptr<SimpleConv>& model_actor, std::shared_ptr<SimpleConv>& model_critic, Hyperparameters& params, int64_t w) {
    // Epsilon is adjusted on a schedule, not fixed
    float epsilon = 0;

    SnakeEnv environment(w,w);
    auto action_space = environment.get_action_space();
    const auto observation_space = environment.get_observation_space();

    cerr << action_space << '\n';

    // Used for deciding if we want to act greedily or sample randomly
    mt19937 generator(1337);
    std::uniform_int_distribution<int64_t> choice_dist(0,4-1);
    std::uniform_int_distribution<int64_t> length_dist(3,6);

    torch::optim::RMSprop optimizer_actor(model_actor->parameters(), torch::optim::RMSpropOptions(params.learn_rate).weight_decay(0.001));
    torch::optim::RMSprop optimizer_critic(model_critic->parameters(), torch::optim::RMSpropOptions(params.learn_rate).weight_decay(0.001));

    auto prev_action = environment.get_action_space();

    size_t e = 0;

    Episode episode;

    while (e < params.n_episodes) {
        environment.reset(length_dist(generator));
        episode.clear();

        // exponential decay that terminates at ~0.018
        // epsilon = pow(0.99,(float(e)/float(params.n_episodes)) * float(size_t(eps_norm))));
        std::bernoulli_distribution dist(epsilon);

        for (size_t s=0; s<params.episode_length; s++) {
            // Construct 1-hot to indicate prev action
            prev_action *= 0;
            prev_action[environment.get_prev_action()] = 1;

            auto input = environment.get_observation_space();
            input += 0.0001;

            // Get value prediction (singleton tensor)
            auto value_predict = model_critic->forward(input);

            // Get action distribution of the policy (shape of action space)
            auto log_probabilities = model_actor->forward(input);
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

            // cerr << "probabilities: \n" << probabilities << '\n';
            // cerr << "choice: " << choice << "\n";
            // cerr << "prev_action: " << environment.get_prev_action() << "\n";
            // cerr << "length: " << episode.get_size() << "\n";
            // cerr << "epsilon: " << epsilon << "\n";

            environment.step(choice);

            float reward = environment.get_reward();

            episode.update(log_probabilities, value_predict, choice, reward);

            if (environment.is_terminated() or environment.is_truncated()) {
                break;
            }
        }

        if (episode.get_size() < 4) {
            continue;
        }
        else {
            e++;
        }

        auto td_loss = - episode.compute_td_loss(params.gamma, false, true);
        auto entropy_loss = - episode.compute_entropy_loss(false, true);

        auto actor_loss = td_loss + params.lambda*entropy_loss;
        auto critic_loss = episode.compute_critic_loss(params.gamma, true);

        // Print some stats, increment loss using episode, update model if batch_size accumulated
        cerr << std::left
        << std::setw(8)  << "episode" << std::setw(8) << e
        << std::setw(8)  << "length" << std::setw(6) << episode.get_size()
        << std::setw(14) << "entropy_loss" << std::setw(12) << entropy_loss.item<float>()*params.lambda
        << std::setw(14) << "avg_entropy" << std::setw(12) << -entropy_loss.item<float>()/float(episode.get_size())
        << std::setw(8)  << "td_loss " << std::setw(12) << td_loss.item<float>()
        << std::setw(14) << "critic_loss" << std::setw(12) << critic_loss.item<float>()
        << std::setw(10) << "epsilon " << std::setw(10) << epsilon << '\n';

        actor_loss.backward(torch::Tensor(), true);
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


void test_actor_critic(std::shared_ptr<SimpleConv>& model_actor, std::shared_ptr<SimpleConv>& model_critic, Hyperparameters& params, int64_t w) {
    // Epsilon is adjusted on a schedule, not fixed
    float epsilon = 0;

    SnakeEnv environment(w,w);
    auto action_space = environment.get_action_space();
    const auto observation_space = environment.get_observation_space();

    cerr << action_space << '\n';

    // Used for deciding if we want to act greedily or sample randomly
    mt19937 generator(1337);
    std::uniform_int_distribution<int64_t> choice_dist(0,4-1);

    torch::optim::RMSprop optimizer_actor(model_actor->parameters(), params.learn_rate);
    torch::optim::RMSprop optimizer_critic(model_critic->parameters(), params.learn_rate);
    auto prev_action = environment.get_action_space();

    size_t e = 0;
    std::thread t(&SnakeEnv::render, &environment, false);

    while (e < params.n_episodes) {
        environment.reset();

        // exponential decay that terminates at ~0.018
        // epsilon = pow(0.99,(float(e)/float(params.n_episodes)) * float(size_t(eps_norm))));
        std::bernoulli_distribution dist(epsilon);

        for (size_t s=0; s<params.episode_length; s++) {
            // Construct 1-hot to indicate prev action
            prev_action *= 0;
            prev_action[environment.get_prev_action()] = 1;

            auto input = environment.get_observation_space().clone();
            input += 0.0001;

            // Get value prediction (singleton tensor)
            auto value_predict = model_critic->forward(input);

            // Get action distribution of the policy (shape of action space)
            auto log_probabilities = model_actor->forward(input);
            auto probabilities = torch::exp(log_probabilities);

            cerr << probabilities << '\n';

            int64_t choice;

            if (dist(generator) == 1) {
                choice = choice_dist(generator);
                cerr << "random" << '\n';
            }
            else {
                choice = torch::argmax(probabilities).item<int64_t>();
                // choice = torch::multinomial(probabilities, 1).item<int64_t>();
            }
            environment.step(choice);

            if (environment.is_terminated() or environment.is_truncated()) {
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }

        e++;
    }

    t.join();
}


void train_policy_gradient(std::shared_ptr<SimpleConv>& model, Hyperparameters& params, int64_t w) {
    // Epsilon is adjusted on a schedule, not fixed
    float epsilon = 0;

    SnakeEnv environment(w,w);
    auto action_space = environment.get_action_space();
    const auto observation_space = environment.get_observation_space();

    cerr << action_space << '\n';

    // Used for deciding if we want to act greedily or sample randomly
    mt19937 generator(1337);
    std::uniform_int_distribution<int64_t> choice_dist(0,4-1);

    torch::optim::RMSprop optimizer(model->parameters(), params.learn_rate);
    auto prev_action = environment.get_action_space();

    size_t e = 0;

    float eps_terminal = 0.01;

    // log_b(x) = log_e(x)/log_e(b)
    float eps_norm = log(0.01)/log(0.99);

    cerr << "Using eps_norm: " << eps_norm << " for eps_terminal: " << eps_terminal << '\n';

    Episode episode;

    while (e < params.n_episodes) {
        environment.reset();
        episode.clear();

        // exponential decay that terminates at ~0.018
        // epsilon = pow(0.99,(float(e)/float(params.n_episodes)) * float(size_t(eps_norm))));
        std::bernoulli_distribution dist(epsilon);

        for (size_t s=0; s<params.episode_length; s++) {
            // Construct 1-hot to indicate prev action
            prev_action *= 0;
            prev_action[environment.get_prev_action()] = 1;

            auto input = environment.get_observation_space().clone();
            input += 0.0001;

            auto log_probabilities = model->forward(input);
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

            // cerr << "probabilities: \n" << probabilities << '\n';
            // cerr << "choice: " << choice << "\n";
            // cerr << "prev_action: " << environment.get_prev_action() << "\n";
            // cerr << "length: " << episode.get_size() << "\n";
            // cerr << "epsilon: " << epsilon << "\n";

            environment.step(choice);

            float reward = environment.get_reward();

            episode.update(log_probabilities, choice, reward);

            if (environment.is_terminated() or environment.is_truncated()) {
                break;
            }
        }

        if (episode.get_size() < 4) {
            continue;
        }
        else {
            e++;
        }

        auto td_loss = episode.compute_td_loss(params.gamma, false, false);
        auto entropy_loss = episode.compute_entropy_loss(false, true);

        // Print some stats, increment loss using episode, update model if batch_size accumulated
        cerr << std::left
        << std::setw(8) << "episode" << std::setw(8) << e
        << std::setw(8) << "length" << std::setw(6) << episode.get_size()
        << std::setw(14) << "entropy_loss" << std::setw(12) << entropy_loss.item<float>()*params.lambda
        << std::setw(8) << "td_loss " << std::setw(12) << td_loss.item<float>()
        << std::setw(10) << "epsilon " << std::setw(10) << epsilon << '\n';

        auto loss = td_loss - params.lambda*entropy_loss;
        loss.backward();

        // Occasionally apply the accumulated gradient to the model
        if (e % params.batch_size == 0){
            cerr << "updating..." << e << '\n';
            optimizer.step();
            optimizer.zero_grad();
        }
    }
}


void test_policy_gradient(std::shared_ptr<SimpleConv>& model, Hyperparameters& params, int64_t w) {
    // TODO: add model.eval!

    // Epsilon is adjusted on a schedule, not fixed
    float epsilon = 0;
    std::bernoulli_distribution dist(epsilon);

    SnakeEnv environment(w,w);
    auto action_space = environment.get_action_space();
    const auto observation_space = environment.get_observation_space();

    cerr << action_space << '\n';

    // Used for deciding if we want to act greedily or sample randomly
    mt19937 generator(1337);
    std::uniform_int_distribution<int64_t> choice_dist(0,4-1);

    vector<float> rewards;
    auto prev_action = environment.get_action_space();

    size_t e = 0;

    std::thread t(&SnakeEnv::render, &environment, false);

    while (e < params.n_episodes) {
        float total_reward = 0;
        environment.reset();
        rewards.clear();

        for (size_t s=0; s<params.episode_length; s++) {
            // Construct 1-hot to indicate prev action
            prev_action *= 0;
            prev_action[environment.get_prev_action()] = 1;

            auto input = environment.get_observation_space().clone();
            input += 0.0001;

            auto log_action = model->forward(input);
            auto probabilities = torch::exp(log_action);

            int64_t choice;

            if (dist(generator) == 1) {
                choice = choice_dist(generator);
            }
            else {
                // choice = torch::argmax(probabilities).item<int64_t>();
                choice = torch::multinomial(probabilities, 1).item<int64_t>();
            }

            environment.step(choice);

            rewards.emplace_back(environment.get_reward());
            total_reward += environment.get_reward();

            std::this_thread::sleep_for(std::chrono::milliseconds(250));

            if (environment.is_terminated() or environment.is_truncated()) {
                break;
            }
        }

        // Print some stats, increment loss using episode, update model if batch_size accumulated
        cerr << "episode=" << e << " step=" << rewards.size() << " total_reward=" << total_reward << " epsilon: " << epsilon << '\n';
    }

    t.join();
}


void train_and_test(Hyperparameters& hyperparams){
    // For now we fix the grid size
    size_t w = 10;

    if (hyperparams.type == "pg") {
        // Input size is the grid (which is flattened) and output size is the action space (for a policy gradient model)
        // Create a new Net. 4 is the actions space size TODO: break out var
        auto model = std::make_shared<SimpleConv>(w, w, 4, 4);

        train_policy_gradient(model, hyperparams, w);
        test_policy_gradient(model, hyperparams, w);
    }
    else if (hyperparams.type == "ac") {
        // Input size is the grid (which is flattened) and output size is the action space (for a policy gradient model)
        // Create a new Net. 4 is the actions space size TODO: break out var
        auto actor = std::make_shared<SimpleConv>(w, w, 4, 4);
        auto critic = std::make_shared<SimpleConv>(w, w, 4, 1);

        train_actor_critic(actor, critic, hyperparams, w);
        test_actor_critic(actor, critic, hyperparams, w);
    }
    else {
        throw runtime_error("ERROR: Unsupported hyperparameter type: " + hyperparams.type);
    }
}


int main(int argc, char* argv[]){
    CLI::App app{"App description"};
    Hyperparameters params;

    app.add_option(
            "--type",
            params.type,
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
        train_and_test(params);
    } CPPTRACE_CATCH(const std::exception& e) {
        std::cerr<<"Exception: "<<e.what()<<std::endl;
        cpptrace::from_current_exception().print_with_snippets();
    }

    return 0;
}
