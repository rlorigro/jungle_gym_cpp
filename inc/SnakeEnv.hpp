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

class SnakeEnv: public Environment{
    torch::Tensor observation_space;
    torch::TensorAccessor<float,3> observation_space_3d;

    torch::Tensor action_space;

    vector<coord_t> xy_permutation;

    mt19937 generator;
    shared_mutex m;

    // The snake will always move forward unless instructed to turn. cached_action is used to cache upcoming action
    // or simply keep track of the prev instruction so the snake continues to move
    atomic<int64_t> cached_action = 0;

    deque <coord_t> snake;
    coord_t apple;

    // auto observation_space_2d = observation_space.accessor<float,2>();

    int64_t width;
    int64_t height;

    int64_t i_permutation = 0;

    static const int64_t UP = 0;
    static const int64_t RIGHT = 1;
    static const int64_t DOWN = 2;
    static const int64_t LEFT = 3;

    static const int64_t SNAKE_BODY = 0;
    static const int64_t SNAKE_HEAD = 1;
    static const int64_t APPLE = 2;
    static const int64_t WALL = 3;

    static constexpr float REWARD_COLLISION = -30;
    static constexpr float REWARD_APPLE = 40;
    static constexpr float REWARD_MOVE = -1;

public:
    void step(const at::Tensor& action);
    void step(int64_t a);
    void step();
    void initialize_snake();
    void fill_wall();
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
