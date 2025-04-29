#pragma once

#include <cstdlib>

namespace JungleGym {

class Hyperparameters {
public:
    size_t episode_length = 16;
    size_t n_episodes = 2000;
    size_t batch_size = 256;
    float learn_rate = 1e-4;
    float learn_rate_final = learn_rate;
    float n_steps = 1'000'000;
    float n_steps_per_cycle = 4096;
    float n_epochs = 4;

    // Decay rate of TD recurrence
    float gamma = 0.95;

    // Weight of entropy term in the loss function
    float lambda = 0.01;

    size_t n_threads = 1;

    // For now just controls the stderr log during training (could be expanded?)
    bool silent = true;

    bool profile = false;
};


}
