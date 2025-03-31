#pragma once

#include <torch/torch.h>
#include <iostream>

std::string get_timestamp() {
    std::ostringstream oss;
    std::time_t t = std::time(nullptr);
    oss << std::put_time(std::localtime(&t), "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}
