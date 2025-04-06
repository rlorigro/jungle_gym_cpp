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

}
