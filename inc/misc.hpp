#pragma once

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdio>
#include <ctime>
#include <utility>

using std::pair;


namespace JungleGym{

using coord_t = pair<int64_t, int64_t>;

std::string get_timestamp();

int64_t nearest_factor(int64_t n, int64_t f);

/**
 * Reset the timestep internal state vars for AdamW in accordance to Adam on Local Time: Addressing Nonstationarity in
 * RL with Relative Adam Timesteps Benjamin Ellis et al.
 * @param optimizer AdamW optimizer to be reset
 */
void reset_adamw_timestep(torch::optim::AdamW& optimizer);


}
