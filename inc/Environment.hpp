#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>
#include <random>
#include <mutex>

using std::mt19937;
using std::mutex;


namespace JungleGym{


class Environment{
public:
    virtual ~Environment() = default;

    virtual void step()=0;
    virtual void reset()=0;
    virtual void render()=0;
    virtual void close()=0;

    float reward;

    // Whether the agent reaches the terminal state (as defined under the MDP of the task) which can be positive or
    // negative. An example is reaching the goal state or moving into the lava from the Sutton and Barto Gridworld
    bool terminated;

    // Whether the truncation condition outside the scope of the MDP is satisfied. Typically, this is a timelimit, but
    // could also be used to indicate an agent physically going out of bounds. Can be used to end the episode
    // prematurely before a terminal state is reached. If true, the user needs to call reset()
    bool truncated;


    at::Tensor action_space;
    at::Tensor observation_space;
    mt19937 generator;
    mutex lock;
};

}

