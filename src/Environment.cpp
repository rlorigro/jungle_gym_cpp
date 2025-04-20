#include "Environment.hpp"

#include <iostream>
#include <stdexcept>

using std::runtime_error;
using std::cerr;


namespace JungleGym{

Environment::Environment():
    total_reward(0),
    reward(0),
    terminated(false),
    truncated(false)
{}

float Environment::get_reward() const{
    return reward;
}

float Environment::get_total_reward() const{
    return total_reward;
}

bool Environment::is_terminated() const{
    return terminated;
}

bool Environment::is_truncated() const{
    return truncated;
}


}
