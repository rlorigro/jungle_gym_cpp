#include "LRScheduler.hpp"
#include <iostream>
#include <stdexcept>

using std::runtime_error;
using std::cerr;


#include <iostream>
#include <cmath>      // std::abs
#include <algorithm>  // std::clamp


bool test_linear_lr_scheduler() {
    const double lr_0 = 0.1;
    const double lr_n = 0.0;
    const size_t n_steps = 10;
    const double tol = 1e-10;

    JungleGym::LinearLRScheduler scheduler(lr_0, lr_n, n_steps);

    for (size_t step = 0; step < n_steps; ++step) {
        double expected_progress = static_cast<double>(step) / static_cast<double>(n_steps - 1);
        double expected_lr = lr_0 * (1.0 - expected_progress) + lr_n * expected_progress;
        expected_lr = std::clamp(expected_lr, lr_n, lr_0);

        double actual_lr = scheduler.next();

        if (std::abs(expected_lr - actual_lr) > tol) {
            std::cerr << "FAIL: Mismatch at step " << step << ":\n";
            std::cerr << "   Expected: " << expected_lr << "\n";
            std::cerr << "   Actual:   " << actual_lr << "\n";
            return false;
        }
    }

    return true;
}


int main(){
    bool success = test_linear_lr_scheduler();

    if (not success) {
      throw std::runtime_error("FAIL");
    }
    else{
        cerr << "PASS" << '\n';
    }

    return 0;
};
