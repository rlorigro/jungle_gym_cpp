#pragma once

#include "Environment.hpp"
#include "misc.hpp"

#include <torch/torch.h>
#include <random>
#include <array>
#include <utility>
#include <memory>
#include <deque>
#include <atomic>
#include <vector>

using std::deque;
using std::shared_ptr;
using std::make_shared;
using std::pair;
using std::mt19937;
using std::array;
using std::atomic;
using std::vector;

namespace JungleGym{

class SnakeEnv: public Environment{
    torch::Tensor observation_space;
    torch::TensorAccessor<float,3> observation_space_3d;

    torch::Tensor action_space;

    vector<coord_t> xy_permutation;

    mutable mt19937 generator;
    shared_mutex m;

    // The snake will always move forward unless instructed to turn. cached_action is used to cache upcoming action
    // or simply keep track of the prev instruction so the snake continues to move
    atomic<int64_t> cached_action = STRAIGHT;

    deque <coord_t> snake;
    coord_t apple;

    // auto observation_space_2d = observation_space.accessor<float,2>();

    int64_t width;
    int64_t height;

    size_t initial_length_min = 4;
    size_t initial_length_max = 6;

    int64_t i_permutation = 0;

    int64_t patience_limit = width*height + 1;
    int64_t patience_counter = 0;

    static constexpr float pi = 3.14159265359;

    static const int64_t LEFT = 0;
    static const int64_t STRAIGHT = 1;
    static const int64_t RIGHT = 2;

    static const int64_t SNAKE_BODY = 0;
    static const int64_t SNAKE_HEAD = 1;
    static const int64_t APPLE = 2;
    static const int64_t WALL = 3;

    static constexpr float REWARD_COLLISION = -1;
    static constexpr float REWARD_APPLE = 3;
    static constexpr float REWARD_MOVE = -0.05;

    void initialize_snake();
    void add_apple_unsafe();
    void fill_wall();
    void add_apple(bool lock);

public:
    void step(int64_t a) override;
    void step();

    [[nodiscard]] bool is_valid(const coord_t& coord) const;
    [[nodiscard]] bool is_open(const coord_t& coord) const;
    void update_coord(int64_t a, coord_t& coord) const;
    void get_head(coord_t& coord) const;
    void get_neck(coord_t& coord) const;

    // This is a factory method, it does not contain any time-dependent information, initialized with zeros
    [[nodiscard]] torch::Tensor get_action_space() const override;
    [[nodiscard]] torch::Tensor get_observation_space() const override;

    void reset() override;
    void render(bool interactive) override;
    void close() override;
    shared_ptr<Environment> clone() const override;

    SnakeEnv(int64_t width, int64_t height);
    SnakeEnv();
};

}
