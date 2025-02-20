#include "cpptrace/from_current.hpp"
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

using namespace JungleGym;

#include <iostream>
#include <stdexcept>

using std::runtime_error;
using std::cerr;


class Hyperparameters {
public:
    size_t episode_length = 400;
    size_t n_episodes = 10000;
    size_t batch_size = 128;
    float gamma = 0.1;
    float learn_rate = 0.001;
};


void discount_rewards(vector<float>& rewards, float gamma){
    float r_prev = 0;
    float avg = 0;

    for (const auto r : rewards) {
        // cerr << r << ',';
        avg += r;
    }
    // cerr << '\n';

    avg /= float(rewards.size());

    // reverse iterate and apply recurrence relation
    for(float& reward : std::ranges::reverse_view(rewards)){
        reward = reward + gamma*r_prev;
        r_prev = reward;
    }

    // for (const auto r : rewards) {
    //     cerr << r << ',';
    // }
    // cerr << '\n';
}


void train_policy_gradient(std::shared_ptr<ShallowNet>& model, Hyperparameters& params, int64_t w) {
    // Epsilon is adjusted on a schedule, not fixed
    float epsilon;

    SnakeEnv environment(w,w);
    auto action_space = environment.get_action_space();
    const auto observation_space = environment.get_observation_space();

    cerr << action_space << '\n';
    cerr << observation_space << '\n';

    // Used for deciding if we want to act greedily or sample randomly
    mt19937 generator(1337);
    std::uniform_int_distribution<int64_t> choice_dist(0,4-1);

    vector<Tensor> log_actions;
    vector<float> rewards;

    torch::optim::SGD optimizer(model->parameters(), params.learn_rate);
    auto prev_action = environment.get_action_space();

    size_t e = 0;

    while (e < params.n_episodes) {
        environment.reset();
        log_actions.clear();
        rewards.clear();

        float total_reward = 0;

        // exponential decay that terminates at ~0.018
        epsilon = pow(0.99,e/(params.n_episodes/400));
        std::bernoulli_distribution dist(epsilon);

        for (size_t s=0; s<params.episode_length; s++) {
            // Construct 1-hot to indicate prev action
            prev_action *= 0;
            prev_action[environment.get_prev_action()] = 1;

            auto input = (torch::cat({environment.get_observation_space().flatten(), prev_action.flatten()}, 0));
            input += 0.001;

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

            // cerr << "probabilities: \n" << probabilities << '\n';
            // cerr << "choice: " << choice << "\n";
            // cerr << "prev_action: " << environment.get_prev_action() << "\n";

            log_actions.emplace_back(log_action[choice]);

            environment.step(choice);

            rewards.emplace_back(environment.get_reward());
            total_reward += environment.get_reward();

            if (environment.is_terminated() or environment.is_truncated()) {
                break;
            }
        }

        if (rewards.size() < 4) {
            continue;
        }
        else {
            e++;
        }

        discount_rewards(rewards, params.gamma);

        // Print some stats, increment loss using episode, update model if batch_size accumulated
        cerr << "episode=" << e << " step=" << rewards.size() << " total_reward=" << total_reward << " epsilon: " << epsilon << '\n';

        auto loss = torch::tensor({1}, torch::kFloat32);

        // Compute gradients for each step and associated reward
        for (size_t i=0; i<log_actions.size(); i++){
            loss += -log_actions[i]*rewards[i];
        }

        loss /= float(params.batch_size);
        loss.backward();

        // Occasionally apply the accumulated gradient to the model
        if (e % params.batch_size == 0){
            cerr << "updating..." << e << '\n';
            optimizer.step();
            optimizer.zero_grad();
        }
    }
}


void test_policy_gradient(std::shared_ptr<ShallowNet>& model, Hyperparameters& params, int64_t w) {
    // Epsilon is adjusted on a schedule, not fixed
    float epsilon = 0;
    std::bernoulli_distribution dist(epsilon);

    SnakeEnv environment(w,w);
    auto action_space = environment.get_action_space();
    const auto observation_space = environment.get_observation_space();

    cerr << action_space << '\n';
    cerr << observation_space << '\n';

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

            auto input = (torch::cat({environment.get_observation_space().flatten(), prev_action.flatten()}, 0));
            input += 0.001;

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

            // TODO: print actions take at end of episode, why does it fail corners?

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

    // Input size is the grid (which is flattened) and output size is the action space (for a policy gradient model)
    // Create a new Net. 4 is the actions space size TODO: break out var
    auto model = std::make_shared<ShallowNet>(w*w + 4, 4);

    train_policy_gradient(model, hyperparams, w);
    test_policy_gradient(model, hyperparams, w);
}


int main(int argc, char* argv[]){
    CLI::App app{"App description"};
    Hyperparameters params;

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
