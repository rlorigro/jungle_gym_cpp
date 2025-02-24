#include "ReplayBuffer.hpp"

#include <ranges>

using std::runtime_error;
using std::cerr;
using std::random_device;
using std::vector;
using std::to_string;
using torch::slice;
using namespace torch::indexing;


namespace JungleGym{

void Episode::clear(){
    log_action_distributions.clear();
    states.clear();
    actions.clear();
    rewards.clear();
    size = 0;
}

size_t Episode::get_size() const{
    return size;
}


void Episode::update(Tensor& log_action_probs, int64_t action_index, float reward){
    // TODO: states
    log_action_distributions.emplace_back(log_action_probs);
    actions.emplace_back(action_index);
    rewards.emplace_back(reward);
    size++;
}


Tensor Episode::compute_entropy(const Tensor& log_distribution) const{
    auto entropy = torch::tensor({0}, torch::dtype(torch::kFloat32).requires_grad(true));;

    for (size_t i=0; i < log_distribution.sizes()[0]; i++){
        entropy = entropy + torch::exp(log_distribution[i]) * log_distribution[i];
    }

    return entropy;
}


Tensor Episode::compute_entropy_loss(bool mean=false) const{
    // Entropy depends on the output of the policy and we use it to compute loss so we require grad here
    auto entropy_loss = torch::tensor({0}, torch::dtype(torch::kFloat32).requires_grad(true));;

    // Compute averaged entropy term for episode action distributions
    // WARNING in-place operators discouraged for libtorch tensors that require grad/autodiff
    for (const auto& dist: log_action_distributions){
        entropy_loss = entropy_loss - compute_entropy(dist);
    }

    if (mean){
        entropy_loss = entropy_loss / log_action_distributions.size();
    }

    return entropy_loss;
}


Tensor Episode::compute_td_loss(float gamma, bool mean=false) const{
    if (gamma < 0 or gamma >=1){
        throw runtime_error("ERROR: gamma must be in range [0,1)");
    }

    // Check that sizes are equal
    if (rewards.size() != log_action_distributions.size()){
        throw runtime_error("ERROR: cannot compute loss for episode with unequal reward/action lengths");
    }

    float r_prev = 0;

    vector<float> td_rewards = rewards;

    // Reverse iterate and apply recurrence relation with gamma
    for(auto& reward: std::ranges::reverse_view(td_rewards)){
        reward = reward + gamma*r_prev;
        r_prev = reward;
    }

    auto loss = torch::tensor({0}, torch::dtype(torch::kFloat32).requires_grad(true));

    // Sum loss for each step using the action probability and the reward according to REINFORCE algorithm
    // WARNING in-place operators discouraged for libtorch tensors that require grad/autodiff
    //
    // NOTE: log probs are negative by default so we subtract because the optimizer step is taken in the direction
    // that MINIMIZES loss.
    for (size_t i=0; i<log_action_distributions.size(); i++){
        loss = loss - log_action_distributions[i][actions[i]]*rewards[i];
    }

    if (mean){
        loss = loss / log_action_distributions.size();
    }

    return loss;
}


}
