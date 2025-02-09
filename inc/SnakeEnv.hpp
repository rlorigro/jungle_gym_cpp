#pragma once

#include "Environment.hpp"

#include <torch/torch.h>
#include <random>
#include <array>
#include <utility>
#include <deque>

using std::deque;
using std::pair;
using std::mt19937;
using std::array;

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
    // TODO: fix render fn
    // TODO: in step function check if should grow
    // TODO: sparsify args for action, don't pass the entire action space...

    at::Tensor observation_space;
    at::Tensor action_space;
    mt19937 generator;
    shared_mutex m;

    deque <pair <int64_t,int64_t> > snake;

    at::TensorAccessor<int32_t,2> observation_space_2d = observation_space.accessor<int32_t,2>();

    int64_t width;
    int64_t height;

public:
    void step(at::Tensor& action);
    void initialize_snake();

    bool is_valid(const coord_t& coord) const;
    bool is_open(const coord_t& coord) const;
    void update_coord(const at::Tensor& action, coord_t& coord) const;
    void get_complement(at::Tensor& action) const;
    int64_t get_complement(int64_t a) const;
    void get_head(coord_t& coord) const;
    void get_neck(coord_t& coord) const;
    
    // This is a factory method, it does not contain any time-dependent information, initialized with zeros
    at::Tensor get_action_space() const;
    const at::Tensor& get_observation_space() const;

    void reset();
    void render();
    void close();

    SnakeEnv(int64_t width, int64_t height);
    SnakeEnv();
};

}
