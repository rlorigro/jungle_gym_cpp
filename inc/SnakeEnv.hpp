#pragma once

#include "Environment.hpp"

#include <torch/torch.h>
#include <random>

using std::mt19937;


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
public:
    void step();
    void reset();
    void render();
    void close();

    float reward;
    bool terminated;
    bool truncated;

    at::Tensor action_space;
    at::Tensor observation_space;

    mt19937 generator;
    mutex lock;

    SnakeEnv(int64_t width, int64_t length);
    SnakeEnv();
};

}
