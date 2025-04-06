#include "torch/torch.h"
#include "misc.hpp"

using torch::Tensor;

#include <iostream>
#include <stdexcept>
#include <random>
#include <vector>

using std::runtime_error;
using std::mt19937;
using std::cerr;
using std::vector;


void test_strided_copy(int64_t factor){
    auto b = torch::arange(32, torch::kInt64);
    b = b.reshape({16,2});

    auto a = b.clone()*0;

    int64_t x = JungleGym::nearest_factor(a.numel(), factor);
    int64_t y = a.numel()/x;

    cerr << "testing: " << factor << ',' << x << ',' << y << '\n';

    auto a_chunks = a.as_strided({y,x},{x,1});
    auto b_chunks = b.as_strided({y,x},{x,1});

    for (int64_t i=0; i<a_chunks.size(0); i++){
        a_chunks[i] = b_chunks[i];
    }

    cerr << "a" << '\n';
    cerr << a << '\n';

    if (not a.equal(b)){
        throw runtime_error("FAIL: a != b");
    }
}


int main(){

    for (int64_t i=1; i<34; i++){
        test_strided_copy(i);
    }

    return 0;
}

