#include <iostream>
#include <stdexcept>

using std::runtime_error;
using std::cerr;

#include <torch/torch.h>
#include "Episode.hpp"

using namespace torch;
using namespace JungleGym;


// Helper to compare tensors approximately
bool approx_equal(const Tensor& a, const Tensor& b, float tol=1e-4) {
    return torch::allclose(a, b, tol);
}


Tensor dummy_log_probs(int64_t action, int64_t num_actions = 3) {
    auto probs = torch::full({num_actions}, -1e9); // log(0) for all
    probs[action] = 0.0; // log(1) for chosen
    return probs;
}


Tensor dummy_value(float value) {
    return torch::tensor({value});
}


Tensor dummy_state(int64_t dim = 4) {
    return torch::zeros({dim});
}


bool test_td_rewards() {
    Episode ep;

    Tensor state;
    Tensor logp;
    Tensor val;

    state = dummy_state();
    logp = dummy_log_probs(0);
    val = dummy_value(0.5);
    ep.update(state, logp, val, 0, 1.0, false, false);

    state = dummy_state();
    logp = dummy_log_probs(1);
    val = dummy_value(0.4);
    ep.update(state, logp, val, 1, 2.0, false, false);

    state = dummy_state();
    logp = dummy_log_probs(2);
    val = dummy_value(0.3);
    ep.update(state, logp, val, 2, 3.0, true, false);

    TensorEpisode te;
    ep.to_tensor(te);

    // ---- test TensorEpisode ----
    cerr << "TensorEpisode" << '\n';

    // gamma = 1
    te.compute_td_rewards(1.0);

    auto expected = torch::tensor({6.0, 5.0, 3.0});
    bool success = approx_equal(te.td_rewards, expected);

    cerr << "expected:\n" << expected << '\n';
    cerr << "te.td_rewards:\n" << te.td_rewards << '\n';

    // ---- test Episode ----
    cerr << "Episode" << '\n';

    // gamma = 1
    vector<float> td_rewards;
    ep.compute_td_rewards(td_rewards, 1.0);

    vector<float> expected_ep = {6.0, 5.0, 3.0};
    success = success and (td_rewards == expected_ep);

    cerr << "expected:\n" << expected_ep << '\n';
    cerr << "te.td_rewards:\n" << td_rewards << '\n';

    return success;
}


bool test_gae() {
    Episode ep;
    Tensor state;
    Tensor logp;
    Tensor val;

    state = dummy_state();
    logp = dummy_log_probs(0);
    val = dummy_value(0.5);
    ep.update(state, logp, val, 0, 1.0, false, false);

    state = dummy_state();
    logp = dummy_log_probs(1);
    val = dummy_value(0.4);
    ep.update(state, logp, val, 1, 2.0, false, false);

    state = dummy_state();
    logp = dummy_log_probs(2);
    val = dummy_value(0.3);
    ep.update(state, logp, val, 2, 3.0, true, false);

    /*
    delta[0] = 1.0 + 0.9 * 0.4 - 0.5 = 0.86
    delta[1] = 2.0 + 0.9 * 0.3 - 0.4 = 1.87
    delta[2] = 3.0 - 0.3             = 2.7

    adv[2] = delta[2] = 2.7
    adv[1] = delta[1] + 0.9 * 0.95 * adv[2] = 1.87 + 0.855 * 2.7   ≈ 4.178
    adv[0] = delta[0] + 0.9 * 0.95 * adv[1] = 0.86 + 0.855 * 4.174 ≈ 4.432

    expected ≈ [4.423, 4.174, 2.7]
    */

    TensorEpisode te;
    ep.to_tensor(te);
    // te.truncation_values = torch::zeros_like(te.rewards);

    auto adv = te.compute_GAE(0.9, 0.95);
    auto expected = torch::tensor({4.432, 4.178, 2.7});

    bool success = approx_equal(adv, expected, 1e-3);

    cerr << "expected:\n" << expected << '\n';
    cerr << "computed:\n" << adv << '\n';

    return success;
}


bool test_gae(float gamma, float lambda, const torch::Tensor& expected) {
    Episode ep;

    Tensor state;
    Tensor logp;
    Tensor val;

    state = dummy_state();
    logp = dummy_log_probs(0);
    val = dummy_value(0.5);
    ep.update(state, logp, val, 0, 1.0, false, false);

    state = dummy_state();
    logp = dummy_log_probs(1);
    val = dummy_value(0.4);
    ep.update(state, logp, val, 1, 2.0, false, false);

    state = dummy_state();
    logp = dummy_log_probs(2);
    val = dummy_value(0.3);
    ep.update(state, logp, val, 2, 3.0, true, false);

    TensorEpisode te;
    ep.to_tensor(te);
//    te.truncation_values = torch::full(te.rewards.sizes(),-Episode::INF);

    auto adv = te.compute_GAE(gamma, lambda);
    bool success = approx_equal(adv, expected, 1e-3);

    cerr << "gamma = " << gamma << ", lambda = " << lambda << '\n';
    cerr << "expected:\n" << expected << '\n';
    cerr << "computed:\n" << adv << '\n';

    return success;
}


bool test_entropy_loss() {
    Episode ep;

    auto log_probs = torch::log_softmax(torch::tensor({1.0, 1.0, 1.0}), 0);
    ep.update(log_probs, 0, 0.0, false, false);
    ep.update(log_probs, 1, 0.0, false, false);

    TensorEpisode te;
    ep.to_tensor(te);

    te.compute_td_rewards(1.0);
    auto entropy = te.compute_entropy_loss(true, true);
    auto expected = torch::tensor(-1.0f);

    // ---- test TensorEpisode ----
    cerr << "TensorEpisode" << '\n';

    bool success = approx_equal(entropy, expected);

    cerr << "expected:\n" << expected << '\n';
    cerr << "entropy:\n" << entropy << '\n';

    // ---- test Episode ----
    cerr << "Episode" << '\n';

    entropy = ep.compute_entropy_loss(true, true);
    success = success and approx_equal(entropy, expected);

    cerr << "expected:\n" << expected << '\n';
    cerr << "entropy:\n" << entropy << '\n';

    return success;
}


bool test_clip_loss() {
    Episode ep;
    Tensor state;
    Tensor logp;
    Tensor val;

    state = dummy_state();
    logp = torch::log(torch::tensor({1.0}));
    val = dummy_value(1.0);
    ep.update(state, logp, val, 0, 1.0, false, false);

    state = dummy_state();
    logp = torch::log(torch::tensor({1.0}));
    val = dummy_value(1.0);
    ep.update(state, logp, val, 0, 1.0, true, false);

    TensorEpisode te;
    ep.to_tensor(te);

    auto new_log_probs = te.log_action_distributions.clone();
    auto clip_loss = te.compute_clip_loss(new_log_probs, 0.0, 0.0, 0.2, true);

    // Should reduce to reward - value
    auto expected_adv = torch::sum(te.rewards - te.value_predictions);

    bool success = approx_equal(clip_loss, expected_adv);

    cerr << "expected:\n" << expected_adv<< '\n';
    cerr << "clip_loss:\n" << clip_loss.item<float>() << '\n';

    te.value_predictions = torch::tensor({2.0, 2.0});
    clip_loss = te.compute_clip_loss(new_log_probs, 0.0, 0.0, 0.2, true);

    // Should be -1.0 if the advantages are -1.0 because the ratio is still 1.0
    // (sign flipped because we minimize loss)
    expected_adv = torch::tensor({1.0});

    success = success and approx_equal(clip_loss, expected_adv);

    cerr << "expected:\n" << expected_adv<< '\n';
    cerr << "clip_loss:\n" << clip_loss.item<float>() << '\n';

    te.value_predictions = torch::tensor({0.0, 0.0});
    clip_loss = te.compute_clip_loss(new_log_probs, 0.0, 0.0, 0.2, true);

    // Should be 1.0 if the advantages are 1.0 because the ratio is still 1.0
    // (sign flipped because we minimize loss)
    expected_adv = torch::tensor({-1.0});

    success = success and approx_equal(clip_loss, expected_adv);

    cerr << "expected:\n" << expected_adv<< '\n';
    cerr << "clip_loss:\n" << clip_loss.item<float>() << '\n';

    new_log_probs = te.log_action_distributions.clone();

    // Divide pi_old by 2
    te.log_action_distributions = te.log_action_distributions - 0.301029996;

    clip_loss = te.compute_clip_loss(new_log_probs, 0.0, 0.0, 0.2, true);

    // Should be clipped at 1.2 if the advantages are 1.0 because the ratio is now 2.0  (1.0/0.5)
    // (sign flipped because we minimize loss)
    expected_adv = torch::tensor({-1.2});

    success = success and approx_equal(clip_loss, expected_adv);

    cerr << "expected:\n" << expected_adv<< '\n';
    cerr << "clip_loss:\n" << clip_loss.item<float>() << '\n';

    return success;
}


bool test_n_episodes() {
    Episode ep;

    auto log_probs = torch::log_softmax(torch::tensor({1.0, 1.0, 1.0}), 0);
    ep.update(log_probs, 0, 0.0, false, false);
    ep.update(log_probs, 1, 0.0, false, true);
    ep.update(log_probs, 1, 0.0, true, false);

    TensorEpisode te;
    ep.to_tensor(te);

    auto n = te.get_n_episodes();
	int64_t expected = 2;

    // ---- test TensorEpisode ----
    cerr << "TensorEpisode" << '\n';

    bool success = (expected == n);

    cerr << "expected:\n" << expected << '\n';
    cerr << "n:\n" << n << '\n';

    // ---- test Episode ----
    cerr << "Episode" << '\n';

    n = ep.get_n_episodes();
    success = success and (expected == n);

    cerr << "expected:\n" << expected << '\n';
    cerr << "n:\n" << n << '\n';

    return success;
}


int main() {
    vector<bool> successes;

    cerr << "-----------------\n";
    cerr << "Test TD Rewards: \n";
    successes.push_back(test_td_rewards());
    cerr << (successes.back() ? "PASS" : "FAIL") << '\n' << '\n';

    cerr << "-----------------\n";
    cerr << "Test GAE:        \n";
    successes.push_back(test_gae());
    cerr << (successes.back() ? "PASS" : "FAIL") << '\n' << '\n';

    cerr << "-----------------\n";
    cerr << "Test GAE (0.9, 0.0)\n";
    successes.push_back(test_gae(0.9, 0.0, torch::tensor({0.86, 1.87, 2.7})));
    cerr << (successes.back() ? "PASS" : "FAIL") << '\n' << '\n';

    cerr << "-----------------\n";
    cerr << "Test GAE (1.0, 1.0)\n";
    successes.push_back(test_gae(1.0, 1.0, torch::tensor({5.5, 4.6, 2.7})));
    cerr << (successes.back() ? "PASS" : "FAIL") << '\n' << '\n';

    cerr << "-----------------\n";
    cerr << "Test GAE (0.0, 0.5)\n";
    successes.push_back(test_gae(0.0, 0.5, torch::tensor({0.5, 1.6, 2.7})));
    cerr << (successes.back() ? "PASS" : "FAIL") << '\n' << '\n';

    cerr << "-----------------\n";
    cerr << "Test Entropy:    \n";
    successes.push_back(test_entropy_loss());
    cerr << (successes.back() ? "PASS" : "FAIL") << '\n' << '\n';

    cerr << "-----------------\n";
    cerr << "Test Clip Loss:  \n";
    successes.push_back(test_clip_loss());
    cerr << (successes.back() ? "PASS" : "FAIL") << '\n' << '\n';

    for (auto success : successes) {
        if (not success) {
            throw runtime_error("FAIL");
        }
    }

    cerr << "all PASS" << '\n';

    return 0;
}

