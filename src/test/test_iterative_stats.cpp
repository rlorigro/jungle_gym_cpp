#include "IterativeStats.hpp"

#include <cassert>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <random>


using std::vector;
using JungleGym::IterativeStats;


// Kind of circular, but whatever, still serves as a reference point for regression
double compute_variance(const std::vector<double>& data) {
    if (data.size() < 2)
        throw std::runtime_error("At least 2 elements required");

    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double accum = 0.0;
    for (double x : data) {
        double diff = x - mean;
        accum += diff * diff;
    }
    return accum / (data.size());
}


// Epsilon for float comparison
constexpr double EPS = 1e-8;

bool test_large_equivalence() {
    constexpr size_t N = 100000;

    IterativeStats stats;
    std::vector<double> values;
    values.reserve(N);

    std::mt19937 rng(1337);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);

    for (size_t i = 0; i < N; ++i) {
        double x = dist(rng);
        values.push_back(x);
        stats.update(x);
    }

    // Ground-truth mean and variance (naive method)
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    double mean = sum / static_cast<double>(N);

    double variance = compute_variance(values);

    // Compare against IterativeStats
    double err_mean = std::abs(stats.get_mean() - mean);
    double err_var = std::abs(stats.get_variance() - variance);

    std::cout << "Mean error: " << err_mean << "\n";
    std::cout << "Variance error: " << err_var << "\n";

    return err_mean < EPS && err_var < EPS;
}


int main(){
    bool success = true;

    std::cerr << "test_large_equivalence" << '\n';
    success = success and test_large_equivalence();

    if (not success) {
        throw std::runtime_error("FAIL");
    }

    return 0;
}
