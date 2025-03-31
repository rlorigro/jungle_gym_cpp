#pragma once

#include <iostream>
#include <stdexcept>
#include <memory>
#include <random>
#include <vector>
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

    vector <vector <size_t> > permutations;
    atomic<size_t> p_index;

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
        options(std::move(options)),
        p_index(0)
{
    // Initialize g moving average
    for (const auto& p : this->params) {
        g.emplace_back(torch::full(p.sizes(), options.g_init, torch::kFloat));
        m.emplace_back(std::make_unique<std::shared_mutex>());
    }

    mt19937 generator((random_device())());

    std::vector<size_t> indices(this->params.size());
    std::iota(indices.begin(), indices.end(), 0);

    cerr << "initializing " << 256 << " permutations of size " << this->params.size() << '\n';
    cerr << std::flush;

    permutations.reserve(256);
    for (size_t i=0; i<256; i++) {
        std::shuffle(indices.begin(), indices.end(), generator);
        permutations.push_back(indices);
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

    mt19937 generator((random_device())());

    std::vector<size_t> indices(this->params.size());
    std::iota(indices.begin(), indices.end(), 0);

    cerr << "initializing " << 256 << " permutations of size " << this->params.size() << '\n';
    cerr << std::flush;

    permutations.reserve(256);
    for (size_t i=0; i<256; i++) {
        std::shuffle(indices.begin(), indices.end(), generator);
        permutations.push_back(indices);
    }
}


void RMSPropAsync::get_params(shared_ptr<torch::nn::Module> model){
    if (model->parameters().size() != params.size()) {
        throw runtime_error("ERROR: RMSPropAsync: wrong number of parameters");
    }

    // Wait for write operation to complete before reading, but allow concurrent read operations
    torch::NoGradGuard no_grad_guard;

    // Get a precomputed shuffled set of parameter indexes for iterating
    auto p = p_index.fetch_add(1);
    p = p % permutations.size();

    const auto& indices = permutations[p];

    for (size_t x=0; x<params.size(); x++){
        auto i = indices[x];
        shared_lock lock(*m[i]);

        model->parameters()[i].copy_(params[i]);
    }
}


void RMSPropAsync::step(const std::vector<Tensor>& worker_params){
    if (worker_params.size() != params.size()) {
        throw runtime_error("ERROR: RMSPropAsync: wrong number of parameters");
    }

    torch::NoGradGuard no_grad_guard;

    float lr = options.lr;
    float e = options.eps;
    float a = options.alpha;

    // Get a precomputed shuffled set of parameter indexes for iterating
    auto p = p_index.fetch_add(1);
    p = p % permutations.size();

    const auto& indices = permutations[p];

    for (size_t x=0; x<worker_params.size(); x++){
        auto i = indices[x];

        // Block all threads from accessing this region of the parameters during writing
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
