#include "Environment.hpp"

#include <iostream>
#include <stdexcept>

using std::runtime_error;
using std::cerr;


namespace JungleGym{

Environment::Environment():
    reward(0),
    terminated(false),
    truncated(false)
{}

float Environment::get_reward() const{
    return reward;
}

bool Environment::is_terminated() const{
    return terminated;
}

bool Environment::is_truncated() const{
    return truncated;
}


}
