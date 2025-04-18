#pragma once

#include <cstdlib>

namespace JungleGym {

class Hyperparameters {
public:
    size_t episode_length = 16;
    size_t n_episodes = 2000;
    size_t batch_size = 16;
    float learn_rate = 5e-5;

    // Decay rate of TD recurrence
    float gamma = 0.7;

    // Weight of entropy term in the loss function
    float lambda = 0.01;

    size_t n_threads = 1;

    // For now just controls the stderr log during training (could be expanded?)
    bool silent = true;

    bool profile = false;
};


}
