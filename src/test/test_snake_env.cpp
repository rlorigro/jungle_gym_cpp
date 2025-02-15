#include "SnakeEnv.hpp"
#include "cpptrace/from_current.hpp"

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


class RandomSnake{
public:
    deque <pair <int64_t,int64_t> > snake;
    mt19937 generator;
    size_t max_width;
    size_t max_height;
    bool dead;
    mutex lock;

    RandomSnake(size_t max_width, size_t max_height);
    RandomSnake(size_t max_width, size_t max_height, int64_t x_start, int64_t y_start);

    bool is_dead() const;
    bool is_self(pair <int64_t, int64_t> coord) const;
    bool is_invalid(pair <int64_t, int64_t> coord) const;
    void move_random();
    void grow_random();
    void grow_random(size_t length);

    // If there is no feasible option this returns false, and updates result the same coord as the snake's head.
    bool random_step(pair<int64_t, int64_t>& result);
};


RandomSnake::RandomSnake(size_t max_width, size_t max_height):
    snake({{0,0}}),
    generator(random_device()()),
    max_width(max_width),
    max_height(max_height),
    dead(false)
{}


RandomSnake::RandomSnake(size_t max_width, size_t max_height, int64_t x_start, int64_t y_start):
    snake({{x_start, y_start}}),
    generator(random_device()()),
    max_width(max_width),
    max_height(max_height),
    dead(false)
{}


bool RandomSnake::is_self(pair <int64_t, int64_t> coord) const {
    for (auto it = snake.begin(); it != snake.end(); ++it) {
        if (*it == coord) {
        return true;
        }
    }

    return false;
}


bool RandomSnake::is_dead() const {
    return dead;
}


bool RandomSnake::is_invalid(pair <int64_t, int64_t> coord) const {
    return coord.first < 0 or coord.first >= max_width or coord.second < 0 or coord.second >= max_height;
}


bool RandomSnake::random_step(pair<int64_t, int64_t>& result) {
    const auto& prev = snake.front();

    if (dead){
        result = prev;
        return false;
    }

    vector<size_t> moves = {0,1,2,3};
    std::shuffle(moves.begin(), moves.end(), generator);

    for (size_t i=0; i <= moves.size(); i++) {
        if (i == moves.size()) {
            result = prev;
            return false;
        }

//        cerr << i << ',' << moves[i] << ',' << prev.first << ',' << prev.second << '\n';

        switch (moves[i]) {
            case 0:
                result.first = prev.first + 1;
                result.second = prev.second;
                break;
            case 1:
                result.second = prev.second + 1;
                result.first = prev.first;
                break;
            case 2:
                result.first = prev.first - 1;
                result.second = prev.second;
                break;
            case 3:
                result.second = prev.second - 1;
                result.first = prev.first;
                break;
            default:
                throw std::runtime_error("ERROR: Invalid range");
        }

//        cerr << '\t' << result.first << ',' << result.second << '\n';

        if (not is_invalid(result) and not is_self(result)) {
            break;
        }
    }

    return true;
}


void RandomSnake::grow_random(size_t length) {
    if (dead){
        return;
    }

    lock.lock();
    pair<int64_t, int64_t> next = snake.front();

    while (snake.size() < length) {
        bool success = random_step(next);

        if (not success) {
            dead = true;
            break;
        }
        snake.emplace_front(next);
    }

    lock.unlock();
}


void RandomSnake::grow_random() {
    if (dead){
        return;
    }

    lock.lock();
    pair<int64_t, int64_t> next = snake.front();

    bool success = random_step(next);
    if (not success) {
        dead = true;
    }

    snake.emplace_front(next);
    lock.unlock();
}


void RandomSnake::move_random() {
    if (dead){
        return;
    }

    snake.pop_back();
    grow_random();
}


void test(){
    size_t w = 10;

    SnakeEnv e(w,w);
    auto action_space = e.get_action_space();

    cerr << action_space << '\n';
    auto action_space_1d = action_space.accessor<int32_t,1>();

    std::thread t(&SnakeEnv::render, &e);

    const auto observation_space = e.get_observation_space();

    cerr << observation_space << '\n';

    mt19937 generator(1337);
    std::uniform_int_distribution<size_t> dist(0, action_space.sizes()[0] - 1); // Create uniform distribution

    for (size_t i=0; i<100; i++) {
        size_t a = dist(generator);

        action_space *= 0;
        action_space_1d[a] = 1;

        e.step(action_space);
        cerr << observation_space << '\n';

        if (e.is_terminated() or e.is_truncated()) {
            e.reset();
        }

        sleep_for(std::chrono::duration<double, std::milli>(250));
    }

    t.join();
}


int main(){
    CPPTRACE_TRY {
        test();
    } CPPTRACE_CATCH(const std::exception& e) {
        std::cerr<<"Exception: "<<e.what()<<std::endl;
        cpptrace::from_current_exception().print_with_snippets();
    }

    return 0;
}
