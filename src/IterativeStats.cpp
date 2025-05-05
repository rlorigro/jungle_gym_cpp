#include "IterativeStats.hpp"
#include <stdexcept>
#include <cmath>


using std::runtime_error;

namespace JungleGym{


IterativeStats::IterativeStats():
        n(0),
        mean(0),
        m2(0)
{}


void IterativeStats::update(double x){
    n++;
    auto delta = x - mean;
    mean += delta/n;
    auto delta2 = x - mean;
    m2 += delta*delta2;
}


double IterativeStats::get_mean() const{
    if (n < 1){
        throw runtime_error("ERROR: IterativeStats::get_mean cannot compute mean on 0 items");
    }

    return mean;
}


double IterativeStats::get_variance() const{
    if (n < 2){
        throw runtime_error("ERROR: IterativeStats::get_variance cannot compute variance on <2 items");
    }

    return m2/n;
}


double IterativeStats::get_stdev() const{
    if (n < 2){
        throw runtime_error("ERROR: IterativeStats::get_stdev cannot compute stdev on <2 items");
    }

    return std::sqrt(m2/n);
}


int64_t IterativeStats::get_n() const{
    return int64_t(round(n));
}


}
