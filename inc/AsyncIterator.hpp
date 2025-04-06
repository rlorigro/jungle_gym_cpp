#pragma once

#include <shared_mutex>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <random>
#include <vector>

#include "torch/torch.h"
#include "misc.hpp"

using std::runtime_error;
using std::random_device;
using std::shared_mutex;
using std::unique_lock;
using std::shared_lock;
using std::unique_ptr;
using std::shared_ptr;
using std::function;
using std::mt19937;
using std::atomic;
using std::vector;
using std::cerr;

using torch::Tensor;

using namespace std::chrono;


namespace JungleGym{


class AsyncIterator{
    vector<Tensor> chunks;

    // For fine grained locking we will access the vectors as chunks using tensor::view(), so we compute the
    // dimensions once and then reuse them later during update. Strides attempt to use the nearest factor to chunk_size
    vector<coord_t> dimensions;

    // Cache a few random permutations to try to keep threads from convoying
    vector <vector<coord_t>> permutations;
    atomic<size_t> p_index;

    vector <vector <unique_ptr<shared_mutex>> > m;

    int64_t chunk_size;

    bool profile;

    atomic<double> wait_time_s;

public:
    inline AsyncIterator(vector<Tensor> data, int64_t chunk_size, int64_t n_threads, bool profile);
    inline AsyncIterator& operator=(const AsyncIterator& other) noexcept;

    AsyncIterator()=default;

    // Read the data from the iterator into a vector of tensors, in a thread-safe way
    inline void read(vector<Tensor>& output);

    // Iterate chunks in a thread-safe way
    inline void for_each_chunk(const function<void(const coord_t& chunk_index, Tensor chunk)>& f);

    // View a vector of tensors in chunks, following the same chunk size as the iterator currently has
    inline void apply_view(vector<Tensor>& output);

    inline double get_wait_time_s() const;
};


double AsyncIterator::get_wait_time_s() const{
    return wait_time_s.load();
}


AsyncIterator& AsyncIterator::operator=(const AsyncIterator& other) {
    if (this != &other) {
        dimensions = other.dimensions;
        permutations = other.permutations;
        p_index = other.p_index.load();
        chunk_size = other.chunk_size;
        profile = other.profile;
        wait_time_s = 0;

        // Create a new set of mutexes (same number as in `other`)
        m.clear();
        for (const auto& mutexes : other.m) {
            std::vector<std::unique_ptr<std::shared_mutex>> new_mutexes;
            for (size_t i = 0; i < mutexes.size(); ++i) {
                new_mutexes.push_back(std::make_unique<std::shared_mutex>());
            }
            m.push_back(std::move(new_mutexes)); // Move into m
        }
    }
    return *this;
}


AsyncIterator::AsyncIterator(vector<Tensor> data, int64_t chunk_size, int64_t n_threads, bool profile):
        chunks(std::move(data)),
        chunk_size(chunk_size),
        profile(profile)
{
    m.reserve(data.size());
    chunks.reserve(data.size());

    // Initialize g moving average
    for (size_t i=0; i<chunks.size(); i++) {
        auto& d = chunks[i];

        // Compute nearest factor (for many smaller tensors this will just default to the numel of the params)
        int64_t x = nearest_factor(d.numel(), chunk_size);
        int64_t y = d.numel()/x;

        dimensions.emplace_back(x,y);
        d = d.view({y,x});

        // Every chunk gets a mutex
        m.emplace_back();
        for (int64_t j=0; j<y; j++) {
            m[i].emplace_back(std::make_unique<std::shared_mutex>());
        }
    }

    mt19937 generator((random_device())());

    std::vector<coord_t> indices;

    for (int64_t a=0; a<dimensions.size(); a++) {
        // Second value is the number of chunks
        for (int64_t b=0; b<dimensions[a].second; b++) {
            indices.emplace_back(a,b);
        }
    }

    // cerr << "total chunks: " << indices.size() << '\n';
    permutations.reserve(n_threads*2);

    for (size_t i=0; i<n_threads*2; i++) {
        std::shuffle(indices.begin(), indices.end(), generator);
        permutations.push_back(indices);
    }

}


void AsyncIterator::read(vector<Tensor>& output){
     auto p = p_index.fetch_add(1);
     p = p % permutations.size();

     const auto& indices = permutations[p];

     auto output_chunks = output;

     // Compute a reshaped version of the params to enable fine-grained locking (match the shape of the parent params)
     for (int64_t i=0; i<output.size(); i++) {
         auto [x,y] = dimensions[i];
         output_chunks[i] = output_chunks[i].view({y,x});
     }

    time_point<high_resolution_clock> t;

     for (size_t i=0; i<indices.size(); i++){
         auto [a,b] = indices[i];

         if (profile) {
             t = high_resolution_clock::now();
         }

         shared_lock lock(*m[a][b]);

         if (profile) {
             wait_time_s += duration<double>(high_resolution_clock::now() - t).count();
         }

         output_chunks[a][b].copy_(chunks[a][b]);
     }
}


void AsyncIterator::for_each_chunk(const function<void(const coord_t& chunk_index, Tensor chunk)>& f){
    // Get a precomputed shuffled set of parameter indexes for iterating
    auto p = p_index.fetch_add(1);
    p = p % permutations.size();

    const auto& indices = permutations[p];

    time_point<high_resolution_clock> t;

    for (size_t i=0; i<indices.size(); i++){
        auto [a,b] = indices[i];

        if (profile) {
            t = high_resolution_clock::now();
        }

        // Block all threads from accessing this region of the parameters during writing
        unique_lock lock(*m[a][b]);

        if (profile) {
            auto elapsed = high_resolution_clock::now() - t;
            wait_time_s += duration<double>(elapsed).count();
        }

        f(indices[i], chunks[a][b]);
    }
}


void AsyncIterator::apply_view(vector<Tensor>& output){
    // Compute a reshaped version of the params to enable fine-grained locking (match the shape of the parent params)
    for (int64_t i=0; i<output.size(); i++) {
        auto [x,y] = dimensions[i];
        output[i] = output[i].view({y,x});
    }
}



}