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
    vector<Tensor> value_predictions;
    vector<Tensor> states;                      // WARNING: UNIMPLEMENTED
    vector<int64_t> actions;
    vector<float> rewards;
    size_t size;

public:
    // Assumes distribution is already in log space and normalized, is either single dimension of length n or 2d with
    // shape [n,1]
    [[nodiscard]] Tensor compute_entropy(const Tensor& log_distribution, bool norm) const;

    // Compute the stepwise entropy for each action distribution and return the sum or mean depending on `mean`
    [[nodiscard]] Tensor compute_entropy_loss(bool mean, bool norm) const;

    // Use the action history and reward history to compute stepwise loss, discounted with decay rate gamma.
    // Return the sum or mean depending on `mean`
    [[nodiscard]] Tensor compute_td_loss(float gamma, bool mean, bool advantage=false, bool terminal=false) const;

    // This corresponds to the MSE of the discounted reward vs predicted value, for each step in retrospect.
    // Return the sum or mean depending on `mean`
    [[nodiscard]] Tensor compute_critic_loss(float gamma, bool mean) const;

    /**
     *
     * @param log_action_probs the log_softmax output distribution for a PG policy at a given time step
     * @param action_index the action chosen at a given time step
     * @param reward the instantaneous reward. This is directly given from the env
     */
    void update(Tensor& log_action_probs, int64_t action_index, float reward);

    /**
     *
     * @param log_action_probs the log_softmax output distribution for a PG policy at a given time step
     * @param value_prediction the predicted value from the critic function
     * @param action_index the action chosen at a given time step
     * @param reward the instantaneous reward. This is directly given from the env
     */
    void update(Tensor& log_action_probs, Tensor& value_prediction, int64_t action_index, float reward);

    Episode():size(0){};

    [[nodiscard]] size_t get_size() const;
    void clear();
};


}