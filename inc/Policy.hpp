#pragma once
#include <torch/torch.h>
#include <vector>
#include <iostream>

using std::vector;
using std::cerr;
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


Tensor ShallowNet::forward(Tensor x){
    // Use one of many tensor manipulation functions.
    x = torch::gelu(layernorm1(fc1->forward(x)));
    x = torch::gelu(layernorm2(fc2->forward(x)));
    x = torch::gelu(layernorm3(fc3->forward(x)));

    // for singleton output, assume no activation needed. e.g. critic value function.
    if (output_size == 1) {
        x = fc4(x);
    }
    else {
        x = torch::log_softmax(fc4->forward(x), 0);
    }

    return x;
}


// convolutional network
class SpatialAttention : public Model {
public:
    int input_width;
    int input_height;

    inline SpatialAttention(int input_width, int input_height, int input_channels);

    inline Tensor forward(Tensor x) override;

    torch::nn::Conv2d conv1{nullptr};
};


SpatialAttention::SpatialAttention(int input_width, int input_height, int input_channels):
    input_width(input_width),
    input_height(input_height)
{
    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, 1, 1).stride(1).padding(0)));
}


Tensor SpatialAttention::forward(Tensor x){
    x = torch::sigmoid(conv1->forward(x));
    return x;
}


class ChannelAttention : public torch::nn::Module {
public:
    // Constructor
    explicit ChannelAttention(int64_t in_channels, int64_t reduction_ratio=2):
        fc1(torch::nn::Linear(in_channels * 2, in_channels / reduction_ratio)),  // First FC layer (reduction)
        fc2(torch::nn::Linear(in_channels / reduction_ratio, in_channels))
    {
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    // Forward pass for channel attention
    torch::Tensor forward(torch::Tensor x) {
        // x shape: [batches, channels, width, height]

        // Perform Global Average Pooling (GAP) and Global Max Pooling (GMP)
        auto gap = torch::mean(x, {2, 3}, true); // Mean across width and height -> [B, C, 1, 1]

        // Max across width (dim=2) and height (dim=3) -> [B, C, 1, 1]
        auto gmp = std::get<0>(torch::max(x, 2, true));      // Max along dim 2 (width)
        gmp = std::get<0>(torch::max(gmp, 3, true));               // Max along dim 3 (height)

        // Flatten the GAP and GMP results (don't remove batch dimension)
        gap = torch::flatten(gap, 1,3);
        gmp = torch::flatten(gmp, 1,3);

        // Concatenate GAP and GMP along channel dimension
        x = torch::cat({gap, gmp}, 1);   // [B, 2*C]

        // Operate on GAP;GMP to get channel weights
        x = torch::gelu(fc1(x));
        x = fc2(x);

        // Apply sigmoid activation to get attention map in range [0, 1]
        auto attention_map = sigmoid(x);  // [B, C]

        // unsqueeze the second and third (W,H) dimensions
        attention_map = attention_map.view({x.size(0), attention_map.size(1), 1, 1});

        return attention_map;
    }

private:
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}; // Fully connected layers
};


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
    torch::nn::Conv2d conv3{nullptr};

    SpatialAttention spatial_map;
    ChannelAttention channel_map;

    torch::nn::Linear fc1{nullptr};
    torch::nn::LayerNorm layernorm1{nullptr};
    torch::nn::Linear fc2{nullptr};
    torch::nn::LayerNorm layernorm2{nullptr};
    torch::nn::Linear fc3{nullptr};
};


SimpleConv::SimpleConv(int input_width, int input_height, int input_channels, int output_size):
    input_width(input_width),
    input_height(input_height),
    output_size(output_size),
    spatial_map(input_width, input_height, input_channels+8+16+32),
    channel_map(input_channels+8+16+32, 2)
{
    int k = 3;

    conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels, 8, k).stride(1).groups(1).padding((k-1)/2)));
    conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels+8, 16, k).stride(1).groups(2).padding((k-1)/2)));
    conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_channels+8+16, 32, k).stride(1).groups(4).padding((k-1)/2)));

    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(input_width*input_height*(input_channels+8+16+32), 512));
    fc2 = register_module("fc2", torch::nn::Linear(512, 256));
    fc3 = register_module("fc3", torch::nn::Linear(256, output_size));

    layernorm1 = register_module("layernorm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({512})));
    layernorm2 = register_module("layernorm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({256})));
}


/**
 * Use the CBAM spatial/channel attention module (without pooling on spatial attention)
 * Use densenet architecture for conv layers
 * @param x input with shape [W,H,C], no batches needed for RL policy
 * @return y
 */
Tensor SimpleConv::forward(Tensor x) {
    // Shape munging into [C,W,H]
    x = torch::permute(x,{2,0,1});

    // Add batch dimension (assumed singleton for RL)
    auto x_input = torch::unsqueeze(x,0);

    // Initial convolutions
    auto x_conv1 = torch::gelu(conv1->forward(x_input));
    x = torch::cat({x_input, x_conv1}, 1);

    auto x_conv2 = torch::gelu(conv2->forward(x));
    x = torch::cat({x_input, x_conv1, x_conv2}, 1);

    auto x_conv3 = torch::gelu(conv3->forward(x));
    x = torch::cat({x_input, x_conv1, x_conv2, x_conv3}, 1);

    // cerr << x_input.sizes() << '\n';
    // cerr << x_conv1.sizes() << '\n';
    // cerr << x_conv2.sizes() << '\n';
    // cerr << x_conv3.sizes() << '\n';
    // cerr << x.sizes() << '\n';

    // Attention mapping
    auto c = x*channel_map.forward(x);
    auto s = c*spatial_map.forward(c);
    x = x + s;

    // Output/prediction layers
    x = torch::flatten(x);
    x = torch::gelu(layernorm1(fc1->forward(x)));
    x = torch::gelu(layernorm2(fc2->forward(x)));

    // for singleton output, assume no activation needed. e.g. critic value function.
    if (output_size == 1) {
        x = torch::gelu(fc3->forward(x));
    }
    else {
        x = torch::log_softmax(fc3->forward(x), 0);
    }
    return x;
}


}