#pragma once

#include <iostream>
#include <stdexcept>
#include <memory>
#include <vector>
#include "torch/torch.h"

using std::runtime_error;
using std::shared_mutex;
using std::unique_lock;
using std::shared_lock;
using std::unique_ptr;
using std::shared_ptr;
using std::vector;
using std::cerr;

using torch::Tensor;
using torch::optim::RMSpropOptions;


namespace JungleGym{


class RMSPropAsyncOptions {
public:
    float lr = 0.0001;
    float alpha = 0.99;
    float eps = 1e-8;

    // Initialize running average with a small non-zero value to prevent large first updates
    float g_init = 10e-3;

    inline RMSPropAsyncOptions(float lr, float alpha, float eps, float g_init);
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


class RMSPropAsync{
    // Parameters to be trained
    vector<Tensor> params;

    // Running elementwise average used for regularizing the updates. Maintained as Exponential Moving Average
    vector<Tensor> g;

    const RMSPropAsyncOptions options;
    vector<unique_ptr<shared_mutex>> m;

public:
    inline RMSPropAsync(vector<Tensor> params, RMSPropAsyncOptions options);
    inline RMSPropAsync(vector<Tensor> params, float lr);
    inline void step(const std::vector<Tensor>& worker_params);
    inline void get_params(shared_ptr<torch::nn::Module> model);
};


RMSPropAsync::RMSPropAsync(vector<Tensor> params, RMSPropAsyncOptions options):
        params(std::move(params)),
        options(std::move(options))
{
    // Initialize g moving average
    for (const auto& p : this->params) {
        g.emplace_back(torch::full(p.sizes(), options.g_init, torch::kFloat));
        m.emplace_back(std::make_unique<std::shared_mutex>());
    }
}


RMSPropAsync::RMSPropAsync(vector<Tensor> params, float lr):
        params(std::move(params)),
        options(lr)
{
    // Initialize g moving average
    for (const auto& p : this->params) {
        g.emplace_back(torch::full(p.sizes(), options.g_init, torch::kFloat));
        m.emplace_back(std::make_unique<std::shared_mutex>());
    }
}


void RMSPropAsync::get_params(shared_ptr<torch::nn::Module> model){
    if (model->parameters().size() != params.size()) {
        throw runtime_error("ERROR: RMSPropAsync: wrong number of parameters");
    }

    // Wait for write operation to complete before reading, but allow concurrent read operations
    // TODO: analyze, check for reader starvation?
    torch::NoGradGuard no_grad_guard;

    for (size_t i = 0; i < params.size(); i++) {
        shared_lock lock(*m[i]);

        model->parameters()[i].copy_(params[i]);
    }
}


void RMSPropAsync::step(const std::vector<Tensor>& worker_params){
    if (worker_params.size() != params.size()) {
        throw runtime_error("ERROR: RMSPropAsync: wrong number of parameters");
    }

    torch::NoGradGuard no_grad_guard;

    // Block all threads from accessing the parameters during writing
    float lr = options.lr;
    float e = options.eps;
    float a = options.alpha;

    for (size_t i=0; i<worker_params.size(); i++){
        unique_lock lock(*m[i]);

        auto& g_wi = worker_params[i].grad();

        auto& g_avg = g[i];
        auto& theta = params[i];

        if (not g_wi.defined()) {
            continue;
        }

        // Imitated from libtorch source
        TORCH_CHECK(!g_wi.is_sparse(), "RMSpropAsync does not support sparse gradients");

        // First apply the gradient
        theta -= lr*(g_wi/(torch::sqrt(g_avg + e)));

        // Then update the g avg
        g_avg *= a;
        g_avg += ((1 - a) * torch::pow(g_wi,2));
    }
}


}
