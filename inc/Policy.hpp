#pragma once
#include <torch/torch.h>
#include <vector>

using std::vector;
using torch::Tensor;

namespace JungleGym{

// Define a new Module.
class Model : public torch::nn::Module {
public:
    // Implement the Net's algorithm.
    virtual Tensor forward(Tensor x)=0;
};


// Define a new Module.
class ShallowNet : public Model {
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
    fc1 = register_module("fc1", torch::nn::Linear(input_size, 256));
    fc2 = register_module("fc2", torch::nn::Linear(256, 256));
    fc3 = register_module("fc3", torch::nn::Linear(256, 256));
    fc4 = register_module("fc4", torch::nn::Linear(256, output_size));

    layernorm1 = register_module("layernorm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({256})));
    layernorm2 = register_module("layernorm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({256})));
    layernorm3 = register_module("layernorm3", torch::nn::LayerNorm(torch::nn::LayerNormOptions({256})));
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


// convolutional network
class SimpleConv : public Model {
public:
    int input_width;
    int input_height;
    int output_size;

    inline SimpleConv(int input_width, int input_height, int input_channels, int output_size);

    inline Tensor forward(Tensor x);

    torch::nn::Conv2d conv1{nullptr};
    torch::nn::Conv2d conv2{nullptr};

    torch::nn::Linear fc1{nullptr};
    torch::nn::LayerNorm layernorm1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::LayerNorm layernorm2{nullptr};
    torch::nn::Linear fc3{nullptr};
};


SimpleConv::SimpleConv(int input_width, int input_height, int input_channels, int output_size):
    input_width(input_width),
    input_height(input_height),
    output_size(output_size)
{
    int k = 5;

    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, 8, k).stride(1).groups(2).padding((k-1)/2)));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 16, k).stride(1).groups(4).padding((k-1)/2)));

    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(input_width*input_height*16, 256));
    fc2 = register_module("fc2", torch::nn::Linear(256, 256));
    fc3 = register_module("fc3", torch::nn::Linear(256, output_size));

    layernorm1 = register_module("layernorm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({256})));
    layernorm2 = register_module("layernorm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({256})));
}


Tensor SimpleConv::forward(Tensor x){
    x = torch::permute(x,{2,0,1});
    x = torch::unsqueeze(x,0);
    x = torch::gelu(conv1->forward(x));
    x = torch::gelu(conv2->forward(x));
    x = torch::flatten(x);
    x = torch::gelu(layernorm1(fc1->forward(x)));
    x = torch::gelu(layernorm2(fc2->forward(x)));
    x = torch::log_softmax(fc3->forward(x), 0);
    return x;
}

//Tensor episode_loss(vector<Tensor> rewards, torch::Op){
//
//}


}