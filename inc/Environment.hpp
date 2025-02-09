#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>
#include <random>
#include <mutex>

using std::shared_mutex;
using std::mt19937;


namespace JungleGym{


class Environment{
protected:
    at::Tensor action_space;
    at::Tensor observation_space;
    float reward;
    bool terminated;
    bool truncated;
    Environment();

public:
    virtual ~Environment() = default;

    // The action tensor is the same dimension as the object returned by action_space()
    virtual void step(at::Tensor& action)=0;

    // This is a factory method, it does not contain any time-dependent information, initialized with zeros
    virtual at::Tensor get_action_space() const=0;

    // This is a factory method, it does not contain any time-dependent information, initialized with zeros
    virtual const at::Tensor& get_observation_space() const=0;

    // The reward given by the current step in the environment
    float get_reward() const;

    // Whether the agent reaches the terminal state (as defined under the MDP of the task) which can be positive or
    // negative. An example is reaching the goal state or moving into the lava from the Sutton and Barto Gridworld
    bool is_terminated() const;

    // Whether the truncation condition outside the scope of the MDP is satisfied. Typically, this is a timelimit, but
    // could also be used to indicate an agent physically going out of bounds. Can be used to end the episode
    // prematurely before a terminal state is reached. If true, the user needs to call reset()
    bool is_truncated() const;

    virtual void reset()=0;
    virtual void render()=0;
    virtual void close()=0;

};

}

