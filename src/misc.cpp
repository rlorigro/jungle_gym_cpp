#include "misc.hpp"

namespace JungleGym{

std::string get_timestamp() {
    std::ostringstream oss;
    std::time_t t = std::time(nullptr);
    oss << std::put_time(std::localtime(&t), "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}


int64_t nearest_factor(int64_t n, int64_t f){
    if (f > n){
        return n;
    }

    for (int64_t i=0; i<f; i++){
        if (n % (f+i) == 0){
            return f+i;
        }
        if (n % (f-i) == 0){
            return f-i;
        }
    }

    return -1;
}


void reset_adamw_timestep(torch::optim::AdamW& optimizer) {
    auto& param_groups = optimizer.param_groups();
    for (auto& group : param_groups) {
        for (auto& p : group.params()) {
            if (not p.grad().defined()) {
                continue;
            }

            // Black magic
            auto& param_state = optimizer.state()[p.unsafeGetTensorImpl()];

            auto& adam_state = static_cast<torch::optim::AdamWParamState&>(*param_state);

            // Reset step to 0
            adam_state.step() = 0;
        }
    }
}


}
