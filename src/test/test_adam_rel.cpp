#include <iostream>
#include <stdexcept>
#include "misc.hpp"

using std::runtime_error;
using std::cerr;
using JungleGym::reset_adamw_timestep;


using namespace torch;

void test_adam_rel(){
    // create a dummy parameter
    Tensor param = torch::randn({3, 3}, torch::requires_grad());
    optim::AdamW optimizer({param}, optim::AdamWOptions(1e-3));

    // run a dummy forward-backward-update step
    Tensor dummy_loss = (param * 2).sum();
    dummy_loss.backward();
    optimizer.step();  // this should increment `step` for the param

    // Step 3: verify that the step is > 0
    {
        auto& state_ptr = optimizer.state().at(param.unsafeGetTensorImpl());
        auto& adam_state = static_cast<optim::AdamWParamState&>(*state_ptr);
        TORCH_CHECK(adam_state.step() > 0, "Initial step was not incremented.");
    }

    // Step 4: call reset function
    reset_adamw_timestep(optimizer);

    // Step 5: verify that the step is reset to zero
    {
        auto& state_ptr = optimizer.state().at(param.unsafeGetTensorImpl());
        auto& adam_state = static_cast<optim::AdamWParamState&>(*state_ptr);
        TORCH_CHECK(adam_state.step() == 0, "Step was not reset to 0.");
    }
}


int main(){
    try{
        test_adam_rel();
    }
    catch (const std::exception& e){
        cerr << "FAIL" << '\n';
        throw e;
    }

    cerr << "PASS" << '\n';

    return 0;
}
