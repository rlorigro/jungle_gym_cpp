#pragma once

#include <cmath>

namespace JungleGym{

class IterativeStats{
    double n;
    double mean;
    double m2;
public:
    IterativeStats();

    void update(double x);
    double get_mean() const;

    // Population variance, not sample
    double get_variance() const;
    double get_stdev() const;
    int64_t get_n() const;
};

}
