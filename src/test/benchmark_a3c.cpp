#include "cpptrace/from_current.hpp"
#include "Hyperparameters.hpp"
#include "Episode.hpp"
#include "SnakeEnv.hpp"
#include "Policy.hpp"
#include "A2CAgent.hpp"
#include "A3CAgent.hpp"
#include "PGAgent.hpp"
#include "CLI11.hpp"
#include "misc.hpp"

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


int main(int argc, char* argv[]){
    path project_dir = (path(__FILE__)).parent_path().parent_path().parent_path();
    path data_dir = project_dir / "data" / "benchmark";

    path initial_output_dir = data_dir / "initial";
    path final_output_dir = data_dir / "final";

    Hyperparameters hyperparams;
    hyperparams.silent = false;

    // Some params that historically work well
    hyperparams.gamma =  0.90;
    hyperparams.learn_rate =  0.0001;
    hyperparams.lambda =  0.07;

    // For now we fix the grid size
    size_t w = 10;

    shared_ptr<SnakeEnv> env = make_shared<SnakeEnv>(w,w);

    int64_t out_size = env->get_action_space().numel();
    shared_ptr<SimpleConv> actor = make_shared<SimpleConv>(w,w,4,out_size);
    shared_ptr<SimpleConv> critic = make_shared<SimpleConv>(w,w,4,1);

//    path initial_output_dir = std::filesystem::weakly_canonical("output") / get_timestamp() / "initial";
//    path final_output_dir = std::filesystem::weakly_canonical("output") / get_timestamp() / "final";

    // --- Train initial ---
    // train very briefly to initialize weights at something not totally insane

//    hyperparams.n_threads = 24;
//    hyperparams.n_episodes = 48;
//
//    A3CAgent a(hyperparams, actor, critic);
//    a.train(env);
//    a.save(initial_output_dir);
//
//    // --- Train full ---
//    // train to point of high average episode length
//
//    hyperparams.n_threads = 24;
//    hyperparams.n_episodes = 50'000;
//
//    A3CAgent b(hyperparams, actor, critic);
//    b.train(env);
//    b.save(final_output_dir);

    cerr << "starting loop over n_threads ..." << '\n';

    cerr.precision(3);

    // --- loop over n_threads ---
    for (auto n: {1,2,4,8,16}) {
        hyperparams.n_threads = n;
        hyperparams.n_episodes = 1000;
        hyperparams.silent = true;
        hyperparams.profile = true;

        // --- Initial ---
        // Measure time to train initial model (short episodes, update is bottleneck)
        A3CAgent c(hyperparams, actor, critic);

        c.load(initial_output_dir / "actor.pt", initial_output_dir / "critic.pt");

        auto start_time = std::chrono::high_resolution_clock::now();
        c.train(env);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = duration<double>(end_time - start_time).count();

        double wait_time_s = c.get_wait_time_s();

        cerr << std::left
             << std::setw(12) << "initial"
             << std::setw(12) << n
             << std::setw(12) << elapsed_time
             << std::setw(12) << wait_time_s/n << "\n";

        // --- Final ---
        // Measure time to train final model (long episodes, update is less of bottleneck)
        A3CAgent d(hyperparams, actor, critic);

        d.load(final_output_dir / "actor.pt", final_output_dir / "critic.pt");

        start_time = std::chrono::high_resolution_clock::now();
        d.train(env);
        end_time = std::chrono::high_resolution_clock::now();
        elapsed_time = duration<double>(end_time - start_time).count();

        wait_time_s = d.get_wait_time_s();

        cerr << std::left
             << std::setw(12) << "final"
             << std::setw(12) << n
             << std::setw(12) << elapsed_time
             << std::setw(12) << wait_time_s/n << "\n";
    }

    return 0;
}
