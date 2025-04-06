#pragma once

#include <shared_mutex>
#include <iostream>
#include <stdexcept>
#include <random>
#include <vector>
#include "AsyncIterator.hpp"
#include "torch/torch.h"

using std::runtime_error;
using std::random_device;
using std::shared_mutex;
using std::unique_lock;
using std::shared_lock;
using std::unique_ptr;
using std::shared_ptr;
using std::vector;
using std::cerr;

using torch::Tensor;
using torch::optim::RMSpropOptions;

using namespace std::chrono;

namespace JungleGym{


class RMSPropAsyncOptions {
public:
    float lr = 0.0001;
    float alpha = 0.99;
    float eps = 1e-8;
    int64_t n_threads = 1;
    int64_t chunk_size = 1024;
    bool profile = false;

    // Initialize running average with a small non-zero value to prevent large first updates
    float g_init = 10e-3;

    inline RMSPropAsyncOptions(float lr, float alpha, float eps, float g_init);
    inline RMSPropAsyncOptions(float lr, bool profile);
    inline RMSPropAsyncOptions(float lr);
};


RMSPropAsyncOptions::RMSPropAsyncOptions(float lr, float alpha, float eps, float g_init):
    lr(lr),
    alpha(alpha),
    eps(eps),
    g_init(g_init)
{}


RMSPropAsyncOptions::RMSPropAsyncOptions(float lr):
    lr(lr)
{}


RMSPropAsyncOptions::RMSPropAsyncOptions(float lr, bool profile):
    lr(lr),
    profile(profile)
{}


class RMSPropAsync{
    // Parameters to be trained
    vector<Tensor> params;

    // Convenience wrapper for fine-grained locking/chunking
    AsyncIterator params_iter;

    // Running elementwise average used for regularizing the updates. Maintained as Exponential Moving Average.
    // Viewed as a shape that enables chunking, same dims as params_iter.
    vector<Tensor> g_chunks;

    const RMSPropAsyncOptions options;

public:
    inline RMSPropAsync(vector<Tensor> params, RMSPropAsyncOptions options);
    inline RMSPropAsync(vector<Tensor> params, float lr, bool profile);
    inline RMSPropAsync(vector<Tensor> params, float lr);
    inline void step(const std::vector<Tensor>& worker_params);
    inline void get_params(shared_ptr<torch::nn::Module> model);
    inline double get_wait_time_s() const;
};


double RMSPropAsync::get_wait_time_s() const{
    return params_iter.get_wait_time_s();
}


RMSPropAsync::RMSPropAsync(vector<Tensor> params, RMSPropAsyncOptions options):
        params(std::move(params)),
        params_iter(this->params, options.chunk_size, options.n_threads, options.profile),
        options(std::move(options))
{
    // Initialize g moving average
    for (const auto& p : this->params) {
        g_chunks.emplace_back(torch::full(p.sizes(), options.g_init, torch::kFloat));
    }

    params_iter.apply_view(g_chunks);
}


RMSPropAsync::RMSPropAsync(vector<Tensor> params, float lr):
    RMSPropAsync(std::move(params), RMSPropAsyncOptions(lr))
{}


RMSPropAsync::RMSPropAsync(vector<Tensor> params, float lr, bool profile):
    RMSPropAsync(std::move(params), RMSPropAsyncOptions(lr, profile))
{}


void RMSPropAsync::get_params(shared_ptr<torch::nn::Module> model){
    if (model->parameters().size() != params.size()) {
        throw runtime_error("ERROR: RMSPropAsync: wrong number of parameters");
    }

    // Wait for write operation to complete before reading, but allow concurrent read operations
    torch::NoGradGuard no_grad_guard;

    auto p = model->parameters();
    params_iter.read(p);
}


void RMSPropAsync::step(const std::vector<Tensor>& worker_params){
    if (worker_params.size() != params.size()) {
        throw runtime_error("ERROR: RMSPropAsync: wrong number of parameters");
    }

    torch::NoGradGuard no_grad_guard;

    float lr = options.lr;
    float e = options.eps;
    float alpha = options.alpha;

    auto g_w = worker_params;
    params_iter.apply_view(g_w);

    params_iter.for_each_chunk([&](const auto& chunk_index, auto theta) {
        auto [a,b] = chunk_index;

        auto& g_wi = g_w[a][b].grad();
        auto g_avg = g_chunks[a][b];

        if (not g_wi.defined()) {
            return;
        }

        // Imitated from libtorch source
        TORCH_CHECK(!g_wi.is_sparse(), "RMSpropAsync does not support sparse gradients");

        // First apply the gradient
        theta -= lr*(g_wi/(torch::sqrt(g_avg + e)));

        // Then update the g avg
        g_avg *= alpha;
        g_avg += (1 - alpha) * torch::pow(g_wi,2);
    });
}


}
