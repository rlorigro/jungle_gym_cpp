#include <vector>
#include <shared_mutex>
#include <mutex>
#include <iostream>
#include <stdexcept>
#include <thread>

using std::vector;
using std::thread;
using std::shared_mutex;
using std::mutex;
using std::unique_lock;
using std::runtime_error;
using std::cerr;

using namespace std::chrono_literals;


void thread_fn(mutex& mutex){
    for (size_t i=0; i<10; i++){
        unique_lock l(mutex);
        std::this_thread::sleep_for(100ms);
    }

    return;
}


int main(){
    mutex m;

    for (size_t n: {1,2,4,8}){
        cerr << n << '\n';
        vector<thread> threads;

        for (size_t i=0; i<n; i++){
            threads.emplace_back(thread_fn, std::ref(m));
        }

        for (auto& t: threads){
            t.join();
        }
    }

    return 0;
};

