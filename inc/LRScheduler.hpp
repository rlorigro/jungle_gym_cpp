#pragma once

#include <torch/torch.h>
#include <algorithm>
#include <stdexcept>

using std::runtime_error;


namespace JungleGym{


class LinearLRScheduler {
    double lr_0;
    double lr_n;
    size_t n_steps;
    size_t i;

public:
    inline LinearLRScheduler(double lr_0, double lr_n, size_t n_steps);
    inline double next();
};


LinearLRScheduler::LinearLRScheduler(double lr_0, double lr_n, size_t n_steps):
    lr_0(lr_0), lr_n(lr_n), n_steps(n_steps), i(0)
{
    if (n_steps <= 1){
        throw runtime_error("ERROR: LinearLRScheduler cannot step with <= 1 step");
    }
}


double LinearLRScheduler::next() {
    double progress = double(i) / double(n_steps - 1);
    double lr = lr_0 * (1.0 - progress) + lr_n * progress;
    lr = std::clamp(lr, lr_n, lr_0);

    i++;

    return lr;
}


}
