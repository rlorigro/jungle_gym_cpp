#include "Episode.hpp"

#include <ranges>

using std::runtime_error;
using std::cerr;
using std::random_device;
using std::vector;
using std::to_string;
using torch::slice;
using namespace torch::indexing;


namespace JungleGym{


TensorEpisode::TensorEpisode(vector<TensorEpisode>& episodes):
        size(0)
{
    vector<Tensor> log_action_distributions_temp;
    vector<Tensor> states_temp;
    vector<Tensor> value_predictions_temp;
    vector<Tensor> actions_temp;
    vector<Tensor> rewards_temp;
    vector<Tensor> mask_temp;
    vector<Tensor> size_temp;

    // This is only a shallow copy, in preparation for torch::cat(vector<Tensor>& t), which is a deep copy
    for (auto& episode : episodes) {
        log_action_distributions_temp.emplace_back(episode.log_action_distributions);
        states_temp.emplace_back(episode.states);
        value_predictions_temp.emplace_back(episode.value_predictions);
        actions_temp.emplace_back(episode.actions);
        rewards_temp.emplace_back(episode.rewards);
        mask_temp.emplace_back(episode.mask);
        size += episode.size;
    }

    log_action_distributions = torch::cat(log_action_distributions_temp);
    states = torch::cat(states_temp);
    value_predictions = torch::cat(value_predictions_temp);
    actions = torch::cat(actions_temp);
    rewards = torch::cat(rewards_temp);
    mask = torch::cat(mask_temp);
}


void TensorEpisode::clear() {
    size = 0;
    td_rewards = torch::Tensor();
}


void TensorEpisode::compute_td_rewards(float gamma){
    td_rewards = torch::zeros(rewards.sizes());

    td_rewards[size-1] = rewards[size-1];

    for (auto i=size-2; i >= 0; i--) {
        td_rewards[i] = gamma*td_rewards[i+1]*mask[i+1] + rewards[i];
    }
}


void Episode::clear(){
    log_action_distributions.clear();
    states.clear();
    value_predictions.clear();
    actions.clear();
    rewards.clear();
    mask.clear();
    size = 0;
}


size_t Episode::get_size() const{
    return size;
}


float Episode::get_total_reward() const {
    float x = 0;
    for (const auto& r: rewards) {
        x += r;
    }

    return x;
}


void Episode::update(Tensor& log_action_probs, int64_t action_index, float reward){
    log_action_distributions.emplace_back(log_action_probs);
    actions.emplace_back(action_index);
    rewards.emplace_back(reward);
    size++;
}


void Episode::update(Tensor& log_action_probs, Tensor& value_prediction, int64_t action_index, float reward){
    log_action_distributions.emplace_back(log_action_probs);
    value_predictions.emplace_back(value_prediction);
    actions.emplace_back(action_index);
    rewards.emplace_back(reward);
    size++;
}


void Episode::update(Tensor& state, Tensor& log_action_probs, Tensor& value_prediction, int64_t action_index, float reward, bool is_reset){
    states.emplace_back(state);
    log_action_distributions.emplace_back(log_action_probs);
    value_predictions.emplace_back(value_prediction);
    actions.emplace_back(action_index);
    rewards.emplace_back(reward);
    mask.emplace_back(!is_reset);
    size++;
}


void Episode::to_tensor(TensorEpisode& tensor_episode) {
    tensor_episode.size = int64_t(size);
    tensor_episode.states = torch::stack(states);
    tensor_episode.log_action_distributions = torch::stack(log_action_distributions);
    tensor_episode.value_predictions = torch::stack(value_predictions);
    tensor_episode.actions = torch::tensor(actions, torch::kInt32);
    tensor_episode.rewards = torch::tensor(rewards, torch::kFloat);
    tensor_episode.mask = torch::tensor(mask, torch::kInt8);
}


Tensor Episode::compute_entropy(const Tensor& log_distribution, bool norm) const{
    auto entropy = torch::tensor({0}, torch::dtype(torch::kFloat32));

    for (size_t i=0; i < log_distribution.sizes()[0]; i++){
        entropy = entropy + torch::exp(log_distribution[i]) * log_distribution[i];
    }

    if (norm){
        // The maximum possible entropy is log(1/A) where A is the size of the action distribution because
        // when the dist is uniform p(x) = 1/A
        auto A = float(log_distribution.sizes()[0]);
        auto denom = torch::log(torch::tensor(1.0f/A));

        // Keep the value negative so that we don't accidentally flip the optimization direction
        entropy = -entropy/denom;
    }

    return entropy;
}


Tensor Episode::compute_entropy_loss(bool mean, bool norm) const{
    // Entropy depends on the output of the policy and we use it to compute loss so we require grad here
    auto entropy_loss = torch::tensor({0}, torch::dtype(torch::kFloat32));

    // Compute averaged entropy term for episode action distributions
    // WARNING in-place operators discouraged for libtorch tensors that require grad/autodiff
    for (const auto& dist: log_action_distributions){
        entropy_loss = entropy_loss - compute_entropy(dist,norm);
    }

    if (mean){
        entropy_loss = entropy_loss / log_action_distributions.size();
    }

    return entropy_loss;
}


Tensor Episode::compute_td_loss(float gamma, bool mean, bool advantage, bool terminal) const{
    if (gamma < 0 or gamma >= 1){
        throw runtime_error("ERROR: gamma must be in range [0,1)");
    }

    // Check that sizes are equal
    if (rewards.size() != log_action_distributions.size()){
        throw runtime_error("ERROR: cannot compute loss for episode with unequal reward/action lengths");
    }
    // Check that sizes are equal
    if (advantage and rewards.size() != value_predictions.size()){
        throw runtime_error("ERROR: cannot compute loss with advantage for episode with unequal reward/value lengths");
    }

    float r_prev = 0;

    // If non-terminal episode, then simply approximate the future reward with the last value function,
    // and do not use last step for training
    if (not terminal and advantage) {
        if (value_predictions.empty()) {
            throw runtime_error("ERROR: cannot compute advantage loss for non-terminal episode without value predictions");
        }
        r_prev = value_predictions.back().item<float>();
    }

    auto stop = int64_t(rewards.size() - terminal);
    auto start = 0;

    vector<float> td_rewards = rewards;

    // Reverse iterate and apply recurrence relation with gamma
    for(int64_t i=stop-1; i>=start; i--){
        td_rewards[i] = td_rewards[i] + gamma*r_prev;
        r_prev = td_rewards[i];
    }

    auto loss = torch::tensor(0, torch::dtype(torch::kFloat32));

    // Sum loss for each step using the action probability and the reward according to REINFORCE algorithm
    // WARNING in-place operators discouraged for libtorch tensors that require grad/autodiff
    //
    // Extreme cases for the two terms of the loss function:
    // a = logπ(a_t∣s_t)
    // b = R
    //
    // p(ai|si) near 1 implies that a is near 0
    // p(ai|si) near 0 implies that a is near -inf
    //
    // b can be large or small depending on reward given
    //
    // We want a and b to be maximized because when the policy is confident, a is near its maximum value of 0
    //
    // For all things we want to maximize, we give them a (-) sign because by default the optimizer minimizes loss
    // i.e. subtracts the gradient w.r.t. loss
    //
    // NOTE: log probs are negative by default so we subtract because the optimizer step is taken in the direction
    // that MINIMIZES loss.
    for (size_t i=start; i<stop; i++){
        if (advantage){
            // If using advantage, we want to train the model to maximize the "advantage" over the expected value
            // of the next state, i.e. Q(s_t, a_t) - V(s_t), both normalizing the reward and encouraging choices
            // that are "advantageous"
            // cerr << "a=" << actions[i] << " L=" << loss.item<float>() << " l=" << log_action_distributions[i][actions[i]].item<float>()*(td_rewards[i] - value_predictions[i].item<float>()) << "\tlog_p=" << log_action_distributions[i][actions[i]].item<float>() << "\tR=" << td_rewards[i] << "\tr=" << rewards[i] << "\tV=" << value_predictions[i].item<float>() << '\n';
            loss = loss - log_action_distributions[i][actions[i]]*(td_rewards[i] - value_predictions[i].item<float>());
        }
        else{
            // cerr << "a=" << actions[i] << " L=" << loss.item<float>() << " l=" << log_action_distributions[i][actions[i]].item<float>()*td_rewards[i] << "\tlog_p=" << log_action_distributions[i][actions[i]].item<float>()  << "\tp=" << exp(log_action_distributions[i][actions[i]].item<float>()) << "\tR=" << td_rewards[i] << "\tr=" << rewards[i] << '\n';
            loss = loss - log_action_distributions[i][actions[i]]*td_rewards[i];
        }
    }
    // cerr << '\n';

    if (mean){
        loss = loss / log_action_distributions.size();
    }

    return loss;
}


Tensor Episode::compute_critic_loss(float gamma, bool mean, bool terminal) const{
    if (gamma < 0 or gamma >= 1){
        throw runtime_error("ERROR: gamma must be in range [0,1)");
    }

    // Check that sizes are equal
    if (rewards.size() != value_predictions.size()){
        throw runtime_error("ERROR: cannot compute critic loss for episode with unequal reward/value_predictions lengths");
    }

    float r_prev = 0;

    // If non-terminal episode, then simply approximate the future reward with the last value function,
    // and do not use last step for training
    if (not terminal) {
        if (rewards.empty()) {
            throw runtime_error("ERROR: cannot compute advantage loss for non-terminal episode without value predictions");
        }
        r_prev = rewards.back();
    }

    auto stop = int64_t(rewards.size() - terminal);
    auto start = 0;

    vector<float> td_rewards = rewards;

    // Reverse iterate and apply recurrence relation with gamma
    for(int64_t i=stop-1; i>=start; i--){
        td_rewards[i] = td_rewards[i] + gamma*r_prev;
        r_prev = td_rewards[i];
    }

    // Initialize loss
    auto loss = torch::zeros({}, torch::dtype(torch::kFloat32));

    // Sum the squared error loss for each time step
    for (size_t i = start; i < stop; i++) {
        // Compute squared loss between the predicted value and the target return
        loss = loss + torch::pow(td_rewards[i] - value_predictions[i], 2) / 2;
    }

    // If requested, average the loss
    if (mean) {
        loss = loss / td_rewards.size();
    }

    return loss;
}


}
