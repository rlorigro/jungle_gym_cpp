#include "cpptrace/from_current.hpp"
#include "SnakeEnv.hpp"
#include "CLI11.hpp"

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


void test(bool interactive){
    size_t w = 10;

    SnakeEnv e(w,w);
    auto action_space = e.get_action_space();

    cerr << action_space << '\n';

    std::thread t(&SnakeEnv::render, &e, interactive);

    auto observation_space = e.get_observation_space();

    cerr << observation_space << '\n';

    mt19937 generator(1337);
    std::uniform_int_distribution<int64_t> dist(0, action_space.sizes()[0] - 1); // Create uniform distribution

    for (size_t i=0; i<100; i++) {
        if (interactive) {
            e.step();
        }
        else {
            int64_t a = dist(generator);
            e.step(a);
        }

        if (e.is_terminated() or e.is_truncated()) {
            e.reset();
        }

        sleep_for(std::chrono::duration<double, std::milli>(500));
    }

    t.join();
}


int main(int argc, char* argv[]){
    CLI::App app{"App description"};
    bool interactive = false;

    // app.add_option(
    //         "--n_threads",
    //         n_threads,
    //         "Maximum number of threads to use");

    app.add_flag("--interactive", interactive, "Use d_min solution to remove all edges not used");

    try{
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    CPPTRACE_TRY {
        test(interactive);
    } CPPTRACE_CATCH(const std::exception& e) {
        std::cerr<<"Exception: "<<e.what()<<std::endl;
        cpptrace::from_current_exception().print_with_snippets();
    }

    return 0;
}
