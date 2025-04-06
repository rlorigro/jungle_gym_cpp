#include "torch/torch.h"
#include "AsyncIterator.hpp"
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

using JungleGym::AsyncIterator;


void test_strided_copy(int64_t factor){
    vector<Tensor> a;
    vector<Tensor> b;

    b.emplace_back(torch::arange(32, torch::kInt64));
    b.back() = b.back().reshape({16,2});

    a.emplace_back(b.back().clone()*0);

    AsyncIterator iter(b, factor, 1, false);

    auto a_chunks = a;
    iter.apply_view(a_chunks);

    iter.for_each_chunk([&](auto chunk_index, auto chunk){
        auto [i,j] = chunk_index;
        a_chunks[i][j] = chunk;
    });

//    cerr << "a" << '\n';
//    cerr << a << '\n';

    for (size_t i=0; i<a.size(); i++){
        if (not a[i].equal(b[i])){
            throw runtime_error("FAIL: a != b");
        }
        else{
            cerr << "PASS" << '\n';
        }
    }

    a.clear();
    a.emplace_back(b.back().clone()*0);

    iter.read(a);

//    cerr << "a" << '\n';
//    cerr << a << '\n';

    for (size_t i=0; i<a.size(); i++){
        if (not a[i].equal(b[i])){
            throw runtime_error("FAIL: a != b");
        }
        else{
            cerr << "PASS" << '\n';
        }
    }
}


int main(){

    for (int64_t i=1; i<34; i++){
        cerr << "testing: " << i << '\n';
        test_strided_copy(i);
    }

    return 0;
}

