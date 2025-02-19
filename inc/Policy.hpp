#pragma once
#include <torch/torch.h>
#include <vector>

using std::vector;
using torch::Tensor;

namespace JungleGym{

// Define a new Module.
class ShallowNet : public torch::nn::Module {
public:
    int input_size;
    int output_size;

    inline ShallowNet(int input_size, int output_size);

    // Implement the Net's algorithm.
    inline Tensor forward(Tensor x);

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr};
    torch::nn::LayerNorm layernorm1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::LayerNorm layernorm2{nullptr};
    torch::nn::Linear fc3{nullptr};
    torch::nn::LayerNorm layernorm3{nullptr};
    torch::nn::Linear fc4{nullptr};
};

ShallowNet::ShallowNet(int input_size, int output_size):
input_size(input_size),
output_size(output_size)
{
    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(input_size, 128));
    fc2 = register_module("fc2", torch::nn::Linear(128, 128));
    fc3 = register_module("fc3", torch::nn::Linear(128, 128));
    fc4 = register_module("fc4", torch::nn::Linear(128, output_size));

    layernorm1 = register_module("layernorm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({128})));
    layernorm2 = register_module("layernorm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({128})));
    layernorm3 = register_module("layernorm3", torch::nn::LayerNorm(torch::nn::LayerNormOptions({128})));
}


// The forward method here does not take batches for RL rollouts...
// Need to check later what shape is compatible with the optim/etc
Tensor ShallowNet::forward(Tensor x){
    // Use one of many tensor manipulation functions.
    x = torch::gelu(layernorm1(fc1->forward(x)));
    x = torch::gelu(layernorm2(fc2->forward(x)));
    x = torch::gelu(layernorm3(fc3->forward(x)));
    x = torch::log_softmax(fc4->forward(x), 0);
    return x;
}

//Tensor episode_loss(vector<Tensor> rewards, torch::Op){
//
//}


}