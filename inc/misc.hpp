#pragma once

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

}
