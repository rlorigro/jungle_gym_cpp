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

const float Episode::INF = std::numeric_limits<float>::infinity();
const float TensorEpisode::INF = std::numeric_limits<float>::infinity();

TensorEpisode::TensorEpisode(vector<TensorEpisode>& episodes):
        size(0)
{
    vector<Tensor> log_action_distributions_temp;
    vector<Tensor> states_temp;
    vector<Tensor> value_predictions_temp;
    vector<Tensor> actions_temp;
    vector<Tensor> rewards_temp;
    vector<Tensor> terminated_temp;
    vector<Tensor> truncated_temp;
    vector<Tensor> size_temp;

    // This is only a shallow copy, in preparation for torch::cat(vector<Tensor>& t), which is a deep copy
    for (auto& episode : episodes) {
        log_action_distributions_temp.emplace_back(episode.log_action_distributions);
        states_temp.emplace_back(episode.states);
        value_predictions_temp.emplace_back(episode.value_predictions);
        actions_temp.emplace_back(episode.actions);
        rewards_temp.emplace_back(episode.rewards);
        terminated_temp.emplace_back(episode.terminated);
        truncated_temp.emplace_back(episode.truncation_values);

        size += episode.size;
    }

    log_action_distributions = torch::cat(log_action_distributions_temp);
    states = torch::cat(states_temp);
    value_predictions = torch::cat(value_predictions_temp);
    actions = torch::cat(actions_temp);
    rewards = torch::cat(rewards_temp);
    terminated = torch::cat(terminated_temp);
    truncation_values = torch::cat(truncated_temp);
}


void TensorEpisode::clear() {
    size = 0;
    td_rewards = Tensor();
}


void TensorEpisode::compute_td_rewards(float gamma) {
    td_rewards = torch::zeros(rewards.sizes(), rewards.options());
    Tensor r = torch::zeros({}, rewards.options());

    for (int64_t t=size-1; t>=0; t--) {
        if (terminated[t].item<bool>()) {
            // Reset to zero at episode boundaries
            r = torch::zeros({}, rewards.options());
        } else {
            auto v = truncation_values[t].item<float>();
            if (v > -INF) {
                // Bootstrap and reset (ignore future values)
                r = truncation_values[t];
            }
        }

        r = rewards[t] + gamma * r;
        td_rewards[t] = r;
    }
}


void TensorEpisode::for_each_batch(int64_t batch_size, const function<void(TensorEpisode& batch)>& f) const {
    int64_t n_batches = size / batch_size;
    int64_t remainder = size % batch_size;

    TensorEpisode e;
    e.size = batch_size;

    for (int64_t i=0; i<n_batches; i++) {
        if (log_action_distributions.defined()){
            e.log_action_distributions = log_action_distributions.slice(0, i*batch_size, (i+1)*batch_size);
        }
        if (value_predictions.defined()){
            e.value_predictions = value_predictions.slice(0, i*batch_size, (i+1)*batch_size);
        }
        else{
            throw runtime_error("ERROR: cannot batch episode without terminated Tensor");
        }
        if (states.defined()){
            e.states = states.slice(0, i*batch_size, (i+1)*batch_size);
        }
        if (actions.defined()){
            e.actions = actions.slice(0, i*batch_size, (i+1)*batch_size);
        }
        if (rewards.defined()){
            e.rewards = rewards.slice(0, i*batch_size, (i+1)*batch_size);
        }
        if (terminated.defined()){
            e.terminated = terminated.slice(0, i*batch_size, (i+1)*batch_size);
        }
        else{
            throw runtime_error("ERROR: cannot batch episode without terminated Tensor");
        }
        if (truncation_values.defined()){
            e.truncation_values = truncation_values.slice(0, i*batch_size, (i+1)*batch_size);
        }
        else{
            throw runtime_error("ERROR: cannot batch episode without truncation_values Tensor");
        }
        if (td_rewards.defined()) {
            e.td_rewards = td_rewards.slice(0, i*batch_size, (i+1)*batch_size);
        }

        bool is_truncated = truncation_values[(i+1)*batch_size-1].item<float>() > -INF;
        bool is_terminated = terminated[(i+1)*batch_size-1].item<bool>();

        // Initialize the t+1 value for the end of the episode batch, because truncated eps must be bootstrapped
        // and here we are artificially truncating the ep into batches
        if (not (is_truncated or is_terminated) and (i+1)*batch_size < size) {
            truncation_values[(i+1)*batch_size-1] = value_predictions[(i+1)*batch_size];
        }

        f(e);
    }

    if (remainder > 0) {
        e.size = remainder;

        if (log_action_distributions.defined()){
            e.log_action_distributions = log_action_distributions.slice(0, n_batches*batch_size, n_batches*batch_size+remainder);
        }
        if (value_predictions.defined()){
            e.value_predictions = value_predictions.slice(0, n_batches*batch_size, n_batches*batch_size+remainder);
        }
        else{
            throw runtime_error("ERROR: cannot batch episode without terminated Tensor");
        }
        if (states.defined()){
            e.states = states.slice(0, n_batches*batch_size, n_batches*batch_size+remainder);
        }
        if (actions.defined()){
            e.actions = actions.slice(0, n_batches*batch_size, n_batches*batch_size+remainder);
        }
        if (rewards.defined()){
            e.rewards = rewards.slice(0, n_batches*batch_size, n_batches*batch_size+remainder);
        }
        if (terminated.defined()){
            e.terminated = terminated.slice(0, n_batches*batch_size, n_batches*batch_size+remainder);
        }
        else{
            throw runtime_error("ERROR: cannot batch episode without terminated Tensor");
        }
        if (truncation_values.defined()){
            e.truncation_values = truncation_values.slice(0, n_batches*batch_size, n_batches*batch_size+remainder);
        }
        else{
            throw runtime_error("ERROR: cannot batch episode without truncation_values Tensor");
        }
        if (td_rewards.defined()) {
            e.td_rewards = td_rewards.slice(0, n_batches*batch_size, n_batches*batch_size+remainder);
        }

        bool is_truncated = truncation_values[n_batches*batch_size+remainder-1].item<float>() > -INF;
        bool is_terminated = terminated[n_batches*batch_size+remainder-1].item<bool>();

        // Initialize the t+1 value for the end of the episode batch, because truncated eps must be bootstrapped
        // and here we are artificially truncating the ep into batches
        if (not (is_truncated or is_terminated) and n_batches*batch_size+remainder < size) {
            truncation_values[n_batches*batch_size+remainder-1] = value_predictions[n_batches*batch_size+remainder];
        }

        f(e);
    }
}


// Compute the stepwise entropy for each action distribution and return the sum or mean depending on `mean`
Tensor TensorEpisode::compute_entropy_loss(bool mean, bool norm) const {
    return compute_entropy_loss(log_action_distributions, mean, norm);
}


// Compute the stepwise entropy for each action distribution and return the sum or mean depending on `mean`
Tensor TensorEpisode::compute_entropy_loss(const Tensor& log_action_probs, bool mean, bool norm) const{
    // actions shape: [N,A] where A is the action space size, N is batch size
    // Compute -Σ log(p(x_i))*p(x_i) for i in A
    // NOTE: distributions are already in log space

    // Avoid -nan if the action space ever approaches 0 (causes crash by torch assertion)
    auto safe_log_probs = log_action_probs.clamp(-100.0, 0.0);

    auto entropy = -torch::sum(torch::exp(safe_log_probs)*safe_log_probs, 1);

    if (norm){
        // The maximum possible entropy is -log(1/A) or log(A) where A is the size of the action distribution because
        // when the dist is uniform p(x) = 1/A
        auto A = float(log_action_probs.sizes()[1]);
        auto denom = torch::log(torch::tensor(A));

        // Keep the value negative so that we don't accidentally flip the optimization direction
        entropy = entropy/denom;
    }

    if (mean) {
        entropy = entropy.mean();
    }
    else {
        entropy = entropy.sum();
    }

    // Negate because this is "loss" and minimizing the negative term is maximizing entropy
    return -entropy;
}


// Use the action history and reward history to compute stepwise loss, discounted with decay rate gamma.
// Return the sum or mean depending on `mean`
Tensor TensorEpisode::compute_td_loss(bool mean, bool advantage) const{
    if (not td_rewards.defined()) {
        throw runtime_error("ERROR: TensorEpisode::compute_td_loss: td_rewards empty, call compute_td_rewards() first!");
    }

    auto loss = torch::tensor({0}, torch::dtype(torch::kFloat32));

    // Shapes:
    // action_distributions  [N,A]
    // actions               [N]    (indexes of chosen actions)
    // td_rewards            [N]
    // value_predicitions    [N]

    auto action_log_probs = log_action_distributions.gather(1, actions.unsqueeze(1));

    if (advantage){
        // If using advantage, we want to train the model to maximize the "advantage" over the expected value
        // of the next state, i.e. Q(s_t, a_t) - V(s_t)

        // Don't want to backprop through the critic so we detach
        loss = torch::sum(action_log_probs*(td_rewards - value_predictions.detach()));
    }
    else{
        loss = torch::sum(action_log_probs*td_rewards);
    }

    if (mean){
        loss = loss / size;
    }

    return -loss;
}


Tensor TensorEpisode::compute_GAE(float gamma, float lambda) const{
    Tensor advantages = torch::zeros({size}, rewards.options()).squeeze();
    Tensor last_advantage = torch::zeros({}, rewards.options());
    Tensor next_value;

    // Initialize the t+1 value for the end of the episode
    if (truncation_values[size-1].item<float>() > -INF) {
        next_value = truncation_values[size-1];
    }
    else if (terminated[size-1].item<bool>()){
        next_value = torch::zeros({}, rewards.options());
    }
    else {
        throw runtime_error("ERROR: last item in TensorEpisode neither terminated nor truncated, cannot bootstrap V");
    }

    // `terminated` is kInt8, shape [N]
    for (int64_t t = size-1; t >= 0; t--) {

        if (t < size - 1) {
            // Is -INF if non-truncated, cached T+1 value otherwise
            if (truncation_values[t].item<float>() > -INF) {
                // Bootstrap and reset (ignore future values)
                next_value = truncation_values[t];
            } else {
                next_value = value_predictions[t+1];
            }
        }

        Tensor delta = rewards[t] + gamma * next_value * (1- terminated[t]) - value_predictions[t];
        advantages[t] = delta + gamma * lambda * (1- terminated[t]) * last_advantage;
        last_advantage = advantages[t];
    }

    return advantages.detach();
}


Tensor TensorEpisode::compute_clip_loss(Tensor& log_action_probs_new, float gae_lambda, float gae_gamma, float eps, bool mean) const{
    // Shapes:
    // action_distributions  [N,A]
    // actions               [N]    (indexes of chosen actions)
    // td_rewards            [N]
    // value_predicitions    [N]

    auto p_old = log_action_distributions.detach().gather(1, actions.unsqueeze(1));
    auto p_new = log_action_probs_new.gather(1, actions.unsqueeze(1));

    // Advantage estimate. Don't want to backprop through the critic so it is detached (see compute_GAE method)
    auto a = compute_GAE(gae_gamma, gae_lambda);
    auto r = torch::exp(p_new - p_old).squeeze();

    auto r_clip = torch::clip(r, 1-eps, 1+eps);

    auto loss = torch::min(r*a, r_clip*a);

    if (mean){
        loss = loss.mean();
    }
    else {
        loss = loss.sum();
    }

    return -loss;
}


// This corresponds to the MSE of the discounted reward vs predicted value, for each step in retrospect.
// Return the sum or mean depending on `mean`
Tensor TensorEpisode::compute_critic_loss(bool mean) const{
    if (not td_rewards.defined()) {
        throw runtime_error("ERROR: TensorEpisode::compute_td_loss: td_rewards empty, call compute_td_rewards() first!");
    }

    auto loss = torch::sum(torch::pow(td_rewards - value_predictions, 2));

    // If requested, average the loss
    if (mean) {
        loss = loss / size;
    }

    return loss;
}


void Episode::clear(){
    log_action_distributions.clear();
    states.clear();
    value_predictions.clear();
    actions.clear();
    rewards.clear();
    terminated.clear();
    truncation_values.clear();
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


void Episode::update(Tensor& log_action_probs, int64_t action_index, float reward, bool is_terminated, bool is_truncated){
    if (not is_truncated) {
        log_action_distributions.emplace_back(log_action_probs);
        actions.emplace_back(action_index);
        rewards.emplace_back(reward);
        terminated.emplace_back(is_terminated);
        truncation_values.emplace_back(torch::full({1}, -INF));
        size++;
    }
}


void Episode::update(Tensor& log_action_probs, Tensor& value_prediction, int64_t action_index, float reward, bool is_terminated, bool is_truncated){
    if (not is_truncated) {
        log_action_distributions.emplace_back(log_action_probs);
        value_predictions.emplace_back(value_prediction);
        actions.emplace_back(action_index);
        rewards.emplace_back(reward);
        terminated.emplace_back(is_terminated);
        truncation_values.emplace_back(torch::full({1}, -INF));
        size++;
    }
    else if (size > 0){
        // Don't include the state that was truncated, but store the value in the previous index in case needed for the
        // GAE estimate or similar, which expects a t+1 estimate

        // For cases where we receive the truncated signal, usually the exact boundary is arbitrary
        // so we just pretend this state wasn't visited and update the values
        truncation_values.back() = value_prediction;
    }
    // ELSE: if the episode is empty AND being truncated, then it is ignored, because it's not necessary.
    // In this unusual case, it would have already been artificially truncated in the previous iteration, so skipping it
    // here is equivalent/redundant.
}


void Episode::update(Tensor& state, Tensor& log_action_probs, Tensor& value_prediction, int64_t action_index, float reward, bool is_terminated, bool is_truncated){
    if (not is_truncated) {
        states.emplace_back(state);
        log_action_distributions.emplace_back(log_action_probs);
        value_predictions.emplace_back(value_prediction.view({1}));
        actions.emplace_back(action_index);
        rewards.emplace_back(reward);
        terminated.emplace_back(is_terminated);
        truncation_values.emplace_back(torch::full({1}, -INF));
        size++;
    }
    else if (size > 0){
        // Don't include the state that was truncated, but store the value in the previous index in case needed for the
        // GAE estimate or similar, which expects a t+1 estimate

        // For cases where we receive the truncated signal, usually the exact boundary is arbitrary
        // so we just pretend this state wasn't visited and update the values
        truncation_values.back() = value_prediction;
    }
    // ELSE: if the episode is empty AND being truncated, then it is ignored, because it's not necessary.
    // In this unusual case, it would have already been artificially truncated in the previous iteration, so skipping it
    // here is equivalent/redundant.
}


void Episode::update_truncated(Tensor& value_prediction){
    // Don't include the state that was truncated, but store the value in the previous index in case needed for the
    // GAE estimate or similar, which expects a t+1 estimate
    truncation_values.back() = value_prediction;
}


void Episode::to_tensor(TensorEpisode& tensor_episode) {
    tensor_episode.size = int64_t(size);
    if (not states.empty()) {
        tensor_episode.states = torch::cat(states, 0);
    }
    if (not log_action_distributions.empty()) {
        tensor_episode.log_action_distributions = torch::stack(log_action_distributions);
    }
    if (not value_predictions.empty()) {
        tensor_episode.value_predictions = torch::cat(value_predictions, 0);
    }
    if (not actions.empty()) {
        tensor_episode.actions = torch::tensor(actions, torch::kInt64);
    }
    if (not rewards.empty()) {
        tensor_episode.rewards = torch::tensor(rewards, torch::kFloat);
    }
    if (not terminated.empty()) {
        tensor_episode.terminated = torch::tensor(terminated, torch::kInt8);
    }
    if (not truncation_values.empty()) {
        tensor_episode.truncation_values = torch::cat(truncation_values, 0);
    }
}


Tensor Episode::compute_entropy(const Tensor& log_distribution, bool norm) const{
    auto entropy = torch::tensor({0}, torch::dtype(torch::kFloat32));
    auto log_distribution_safe = log_distribution.clamp(-100,0);

    entropy = torch::sum(torch::exp(log_distribution_safe) * log_distribution_safe);

    if (norm){
        // The maximum possible entropy is -log(1/A) or log(A) where A is the size of the action distribution because
        // when the dist is uniform p(x) = 1/A
        auto A = float(log_distribution.sizes()[0]);
        auto denom = torch::log(torch::tensor(A));

        entropy = entropy/denom;
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

    return -entropy_loss;
}


void Episode::compute_td_rewards(vector<float>& result, float gamma) const{
    result = rewards;
    float r = 0;

    for (int64_t t=int64_t(size)-1; t>=0; t--) {
        if (terminated[t]) {
            // Reset at episode boundaries
            r = 0;
        } else {
            auto v = truncation_values[t].item<float>();
            if (v > -INF) {
                // Bootstrap and reset (ignore future values)
                r = v;
            }
        }

        r = rewards[t] + gamma*r;
        result[t] = r;
    }
}


Tensor Episode::compute_td_loss(float gamma, bool mean, bool advantage) const{
    if (gamma < 0 or gamma >= 1){
        throw runtime_error("ERROR: gamma must be in range [0,1)");
    }

    // Check that sizes are equal
    if (rewards.size() != log_action_distributions.size()){
        throw runtime_error("ERROR: cannot compute loss for episode with unequal reward/action lengths");
    }
    if (advantage and rewards.size() != value_predictions.size()){
        throw runtime_error("ERROR: cannot compute loss with advantage for episode with unequal reward/value lengths: " + to_string(rewards.size()) + "/" + to_string(value_predictions.size()));
    }
    if (terminated.size() != value_predictions.size()){
        throw runtime_error("ERROR: cannot compute loss for episode with unequal terminated/value lengths");
    }
    if (advantage and truncation_values.size() != value_predictions.size()){
        throw runtime_error("ERROR: cannot compute loss with advantage for episode with unequal truncation_values/value lengths");
    }

    vector<float> td_rewards;
    compute_td_rewards(td_rewards, gamma);

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
    for (size_t i=0; i<size; i++){
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


Tensor Episode::compute_critic_loss(float gamma, bool mean) const{
    if (gamma < 0 or gamma >= 1){
        throw runtime_error("ERROR: gamma must be in range [0,1)");
    }

    // Check that sizes are equal
    if (rewards.size() != value_predictions.size()){
        throw runtime_error("ERROR: cannot compute critic loss for episode with unequal reward/value_predictions lengths");
    }
    if (rewards.size() != log_action_distributions.size()){
        throw runtime_error("ERROR: cannot compute critic loss for episode with unequal reward/action lengths");
    }
    if (terminated.size() != value_predictions.size()){
        throw runtime_error("ERROR: cannot compute critic loss for episode with unequal terminated/value lengths");
    }
    if (truncation_values.size() != value_predictions.size()){
        throw runtime_error("ERROR: cannot compute critic loss for episode with unequal truncation_values/value lengths");
    }

    vector<float> td_rewards;
    compute_td_rewards(td_rewards, gamma);

    // Initialize loss
    auto loss = torch::zeros({}, torch::dtype(torch::kFloat32));

    // Sum the squared error loss for each time step
    for (size_t i=0; i<size; i++) {
        // Compute squared loss between the predicted value and the target return
        loss = loss + torch::pow(td_rewards[i] - value_predictions[i], 2);
    }

    // If requested, average the loss
    if (mean) {
        loss = loss / td_rewards.size();
    }

    return loss;
}


}
