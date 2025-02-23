#pragma once

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

using torch::Tensor;

namespace JungleGym{

class Episode{
    vector<Tensor> log_action_distributions;
    vector<Tensor> states;                      // WARNING: UNIMPLEMENTED
    vector<int64_t> actions;
    vector<float> rewards;
    size_t size;

public:
    // Assumes distribution is already in log space and normalized, is either single dimension of length n or 2d with
    // shape [n,1]
    Tensor compute_entropy(const Tensor& log_distribution) const;

    // Compute the stepwise entropy for each action distribution and return the sum or mean depending on `mean`
    Tensor compute_entropy_loss(bool mean) const;

    // Use the action history and reward history to compute stepwise loss, discounted with decay rate gamma.
    // Return the sum or mean depending on `mean`
    Tensor compute_td_loss(float gamma, bool mean) const;

    void update(Tensor& log_action_probs, int64_t action_index, float reward);

    Episode():size(0){};

    size_t get_size() const;
    void clear();
};


}