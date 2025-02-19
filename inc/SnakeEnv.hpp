#pragma once

#include "Environment.hpp"

#include <torch/torch.h>
#include <random>
#include <array>
#include <utility>
#include <deque>
#include <atomic>
#include <vector>

using std::deque;
using std::pair;
using std::mt19937;
using std::array;
using std::atomic;
using std::vector;

using coord_t = pair<int64_t, int64_t>;

namespace JungleGym{

/*

    RandomSnake needs to be refactored into an Environment where:
        is_dead = terminated
        random_step is refactored to fit into the action space sampling procedure?

    Connect terminated criteria to render loop

    Make observation and action space private and only const accessible through getter functions

    Enforce locks on getters/setters using shared mutex, e.g.:

        void read_tensor(const at::Tensor& tensor) {
            std::shared_lock<std::shared_mutex> lock(tensor_mutex);  // Shared lock for reading
            std::cout << "Reading tensor: " << tensor << std::endl;
        }

        void write_tensor(at::Tensor& tensor) {
            std::unique_lock<std::shared_mutex> lock(tensor_mutex);  // Exclusive lock for writing
            tensor[0] = 5;  // Example write operation
            std::cout << "Writing tensor: " << tensor << std::endl;
        }

    ...because tensor read operations are not safe when also writing.
 */


class SnakeEnv: public Environment{
    // TODO: rework encoding for head/body/apple .. just use 1-hot? also head unimplemented currently

    torch::Tensor observation_space;
    torch::Tensor action_space;

    vector<int64_t> x_permutation;
    vector<int64_t> y_permutation;

    mt19937 generator;
    shared_mutex m;

    // The snake will always move forward unless instructed to turn. cached_action is used to cache upcoming action
    // or simply keep track of the prev instruction so the snake continues to move
    atomic<int64_t> cached_action = 0;

    deque <coord_t> snake;
    coord_t apple;

    at::TensorAccessor<float,2> observation_space_2d = observation_space.accessor<float,2>();

    int64_t width;
    int64_t height;

    int64_t i_x = 0;
    int64_t i_y = 0;

    static const int64_t UP = 0;
    static const int64_t RIGHT = 1;
    static const int64_t DOWN = 2;
    static const int64_t LEFT = 3;
    static const int64_t SNAKE_BODY = 1;
    static const int64_t SNAKE_HEAD = 2;
    static const int64_t APPLE = -1;
    static const int64_t REWARD_COLLISION = -10;
    static const int64_t REWARD_APPLE = 5;
    static const int64_t REWARD_MOVE = 1;

public:
    void step(const at::Tensor& action);
    void step(int64_t a);
    void step();
    void initialize_snake();
    void add_apple(bool lock);
    void add_apple_unsafe();

    bool is_valid(const coord_t& coord) const;
    bool is_open(const coord_t& coord) const;
    void update_coord(int64_t a, coord_t& coord) const;
    void get_complement(at::Tensor& action) const;
    int64_t get_complement(int64_t a) const;
    void get_head(coord_t& coord) const;
    void get_neck(coord_t& coord) const;
    int64_t get_prev_action() const;

    // This is a factory method, it does not contain any time-dependent information, initialized with zeros
    torch::Tensor get_action_space() const;
    const torch::Tensor& get_observation_space() const;

    void reset();
    void render(bool interactive);
    void close();

    SnakeEnv(int64_t width, int64_t height);
    SnakeEnv();
};

}
