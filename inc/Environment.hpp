#pragma once

#include <string>
#include <vector>
#include <torch/torch.h>
#include <random>
#include <mutex>
#include <memory>

using std::shared_ptr;
using std::make_shared;
using std::shared_mutex;
using std::mt19937;
using std::atomic;


namespace JungleGym{


class Environment{
protected:
    torch::Tensor action_space;
    torch::Tensor observation_space;
    float reward;
    atomic<bool> terminated;
    atomic<bool> truncated;
    Environment();

public:
    virtual ~Environment() = default;

    // The action tensor is the same dimension as the object returned by action_space()
    virtual void step(int64_t choice)=0;

    // This is a factory method, it does not contain any time-dependent information, initialized with zeros
    virtual torch::Tensor get_action_space() const=0;

    // This is a factory method, it does not contain any time-dependent information, initialized with zeros
    virtual torch::Tensor get_observation_space() const=0;

    // The reward given by the current step in the environment
    float get_reward() const;

    // Whether the agent reaches the terminal state (as defined under the MDP of the task) which can be positive or
    // negative. An example is reaching the goal state or moving into the lava from the Sutton and Barto Gridworld
    bool is_terminated() const;

    // Whether the truncation condition outside the scope of the MDP is satisfied. Can be used to end the episode
    // prematurely before a terminal state is reached. If true, the user needs to call reset(). For the purpose
    // of this library, this flag DOES NOT cover time limits, as it is cleaner to indicate that separately as an agent
    // hyperparameter for algos like A3C which rely on short time limits.
    bool is_truncated() const;

    virtual void reset()=0;
    virtual void render(bool interactive)=0;
    virtual void close()=0;

    // Needed for copying in dynamic dispatch
    virtual shared_ptr<Environment> clone() const=0;

};

}

