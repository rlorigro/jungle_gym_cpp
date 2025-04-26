#pragma once

#include <torch/torch.h>
#include <functional>
#include <utility>
#include <random>
#include <vector>
#include <atomic>
#include <deque>
#include <array>

using std::function;
using std::mt19937;
using std::atomic;
using std::vector;
using std::array;
using std::deque;
using std::pair;

using torch::Tensor;


namespace JungleGym {

/**
 * Similar to Episode, but the time dimension is accounted for in the tensor, instead of using a vector. Intended for
 * batched inference and training with importance sampling (e.g. in PPO)
 */
class TensorEpisode{
public:
    // Warning: all vars defined here must be cleared in the Episode::clear() method!
    Tensor log_action_distributions;
    Tensor value_predictions;
    Tensor states;
    Tensor actions;
    Tensor rewards;
    Tensor td_rewards;

    // 0 by default, 1 on terminal/truncation steps. Used for PPO and recurrent policy implementations.
    Tensor mask;
    int64_t size;

    void clear();
    void compute_td_rewards(float gamma);
    void for_each_batch(int64_t batch_size, const function<void(TensorEpisode& batch)>& f);

    TensorEpisode():size(0){};

    // Compute the stepwise entropy for each action distribution and return the sum or mean depending on `mean`
    [[nodiscard]] Tensor compute_entropy_loss(const Tensor& log_action_probs, bool mean, bool norm) const;
    [[nodiscard]] Tensor compute_entropy_loss(bool mean, bool norm) const;

    // Use the action history and reward history to compute stepwise loss, discounted with decay rate gamma.
    // Return the sum or mean depending on `mean`
    [[nodiscard]] Tensor compute_td_loss(bool mean, bool advantage) const;

    // This corresponds to the MSE of the discounted reward vs predicted value, for each step in retrospect.
    // Return the sum or mean depending on `mean`
    [[nodiscard]] Tensor compute_critic_loss(bool mean) const;

    /**
     * Assuming that this TensorEpisode has action probabilities stored from θ_old, compute the L_CLIP loss term as
     * defined by PPO (Schulman et al. 2017) in which the ratio and the gradients are computed using θ_new provided
     * by the user during batched training (must be in log space). Also assumes critic's value estimates are assigned to
     * this instance's value_predictions attribute.
     * @param mean Whether to average the clip loss over a batch
     * @param log_action_probs_new the output of the model being trained θ_new as in PP. Shape: [N,A], batched and log
     * @return loss Tensor for computing gradients
     */
    [[nodiscard]] Tensor compute_clip_loss(Tensor& log_action_probs_new, float eps, bool mean) const;

    /**
     * Construct a TensorEpisode as a concatenation of any number of others
     * @param episodes vector of TensorEpisodes to be concatenated into this one
     */
    TensorEpisode(vector<TensorEpisode>& episodes);
};


/**
 * Used for stepwise accumulation of training rollouts. Simple vectors maintain mutability.
 **/
class Episode{
    // Warning: all vars defined here must be cleared in the Episode::clear() method!
    vector<Tensor> log_action_distributions;
    vector<Tensor> value_predictions;
    vector<Tensor> states;
    vector<int64_t> actions;
    vector<float> rewards;

    // 1 by default, 0 on first step of new episodes. Used for PPO and recurrent policy implementations.
    vector<int8_t> mask;
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
    [[nodiscard]] Tensor compute_critic_loss(float gamma, bool mean, bool terminal) const;

    float get_total_reward() const;

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

    /**
     *
     * @param state the state which led to the outputs in this update
     * @param log_action_probs the log_softmax output distribution for a PG policy at a given time step
     * @param value_prediction the predicted value from the critic function
     * @param action_index the action chosen at a given time step
     * @param reward the instantaneous reward. This is directly given from the env
     * @param is_reset whether the episode has been reset before this step (i.e. is this the first step)
     */
    void update(Tensor& state, Tensor& log_action_probs, Tensor& value_prediction, int64_t action_index, float reward, bool is_reset);

    void to_tensor(TensorEpisode& tensor_episode);

    Episode():size(0){};

    [[nodiscard]] size_t get_size() const;
    void clear();
};


}
