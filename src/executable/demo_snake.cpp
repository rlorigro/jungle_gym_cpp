#include <iostream>
#include <stdexcept>

using std::runtime_error;
using std::cerr;

#include "cpptrace/from_current.hpp"
#include "Hyperparameters.hpp"
#include "IterativeStats.hpp"
#include "Episode.hpp"
#include "SnakeEnv.hpp"
#include "Policy.hpp"
#include "A2CAgent.hpp"
#include "A3CAgent.hpp"
#include "PPOAgent.hpp"
#include "PGAgent.hpp"
#include "CLI11.hpp"
#include "misc.hpp"

#include "gif.h"

// This must be included to properly implement mimalloc
#include "mimalloc_conditional_override.hpp"

#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <utility>
#include <ranges>
#include <thread>
#include <random>
#include <string>
#include <vector>
#include <deque>
#include <array>

using std::this_thread::sleep_for;
using std::runtime_error;
using std::random_device;
using std::filesystem::exists;
using std::filesystem::create_directories;
using std::mt19937;
using std::vector;
using std::string;
using std::deque;
using std::array;
using std::mutex;
using std::cerr;
using std::pair;

using namespace JungleGym;

using std::runtime_error;
using std::cerr;
using std::min;
using std::to_string;


#define GIF_TEMP_FRAME 0  // no temp frame


/**
* Assume rgba_frames is [C,H,W] where C=4
*/
void save_gif(vector<Tensor> rgba_frames, path out_path, int64_t scale_factor=64, int delay=20) {
    if (rgba_frames.empty()){
        throw runtime_error("ERROR: cannot render empty episode");
    }

    int height = rgba_frames[0].sizes()[1]*scale_factor;
    int width = rgba_frames[0].sizes()[2]*scale_factor;

    cerr << rgba_frames[0].sizes() << '\n';

    GifWriter writer = {};
    string out_path_str = out_path.string();

    GifBegin(&writer, out_path_str.c_str(), width, height, delay);

    for (const auto& frame: rgba_frames) {
        // Resize the frame using nearest-neighbor interpolation (expects [N,C,H,W])
        auto f = torch::nn::functional::interpolate(
            frame.unsqueeze(0),
            torch::nn::functional::InterpolateFuncOptions()
                .scale_factor(std::vector<double>{(double)scale_factor, (double)scale_factor})
                .mode(torch::kNearest))
            .squeeze(0);

        // Ensure tensor is contiguous and on CPU and back into H,W,C (expected by GIF writer)
        f = f.permute({1,2,0});
        f = f.to(torch::kCPU).contiguous();

        // Assume many-hot 0-1 scaled tensor needs to be 255 (8bit)
        f *= 255;

        // Copy pixel data into std::vector<uint8_t>
        std::vector<uint8_t> rgba_frame(width * height * 4);
        f = f.to(torch::kUInt8);

//        auto f_a = f.accessor<uint8_t, 3>();  // shape: [H, W, 4]
//
//        for (int h = 0; h < height; ++h) {
//            for (int w = 0; w < width; ++w) {
//                for (int c = 0; c < 4; ++c) {
//                    rgba_frame[(h * width + w) * 4 + c] = f_a[h][w][c];
//                }
//            }
//        }

        std::memcpy(rgba_frame.data(), f.data_ptr(), rgba_frame.size());

        // Write the frame to the GIF
        GifWriteFrame(&writer, rgba_frame.data(), width, height, delay);
    }

    GifEnd(&writer);
}


void sample_trajectories(shared_ptr<const Environment> env, shared_ptr<SimpleConv> actor, path output_dir){
    if (!env) {
        throw std::runtime_error("ERROR: Environment pointer is null");
    }

    output_dir = std::filesystem::weakly_canonical(output_dir);

    if (not exists(output_dir)) {
        create_directory(output_dir);
    }
    else{
        throw runtime_error("ERROR: output_dir must not exist: " + output_dir.string());
    }

    torch::NoGradGuard no_grad;
    actor->eval();

    shared_ptr<Environment> environment = env->clone();

    vector<Tensor> episode;
    IterativeStats reward_stats;

    size_t n_saved = 0;
    size_t n_burn_in = 100;
    size_t max_demos = 10;

    while (true) {
        environment->reset();
        episode.clear();

        while (true) {
            auto input = environment->get_observation_space();

            // Get action distribution of the policy (shape of action space)
            auto log_probabilities = actor->forward(input);
            auto probabilities = torch::exp(log_probabilities);

            int64_t choice = torch::argmax(probabilities).item<int64_t>();
            // int64_t choice = torch::multinomial(probabilities, 1).item<int64_t>();

            environment->step(choice);
            float total_reward = environment->get_total_reward();
            int64_t n = reward_stats.get_n();

            if (n > int64_t(n_burn_in)) {
                episode.emplace_back(input);
            }

            if (environment->is_terminated() or environment->is_truncated()) {
                if (n > int64_t(n_burn_in)) {
                    auto mean = float(reward_stats.get_mean());
                    auto stdev = float(reward_stats.get_stdev());
                    if (total_reward > mean + 1*stdev){
                        path out_path = output_dir / (to_string(n_saved) + ".gif");

                        cerr << "Saving: " << out_path << '\n';
                        cerr << "total_reward: " << total_reward << '\n';
                        cerr << "length: " << episode.size() << '\n';

                        vector<Tensor> frames;

                        for (auto& state: episode){
                            frames.emplace_back(environment->render_frame(state));
                        };

                        save_gif(frames, out_path);

                        n_saved++;
                    }
                }
                else{
                    reward_stats.update(double(environment->get_total_reward()));
                }

                if (n % 10 == 0 and n > 2) {
                    cerr << n << '\n';
                    cerr << "this run: " << environment->get_total_reward() << '\n';
                    cerr << "mean: " << reward_stats.get_mean() << '\n';
                    cerr << "stdev: " << reward_stats.get_stdev() << '\n';
                }

                break;
            }
        }

        if (n_saved == max_demos){
            break;
        }
    }
}


void demo(path model_dir, path output_dir){
    Hyperparameters hyperparams;
    hyperparams.silent = false;

    // For now we fix the grid size
    size_t w = 10;

    shared_ptr<SnakeEnv> env = make_shared<SnakeEnv>(w,w);

    int64_t out_size = env->get_action_space().numel();
    cerr << "action space: " << out_size << '\n';
    shared_ptr<SimpleConv> actor = make_shared<SimpleConv>(w,w,4,out_size);
    torch::load(actor, model_dir/"actor.pt");

    sample_trajectories(env, actor, output_dir);
}


int main(int argc, char* argv[]){
    CLI::App app{"App description"};
    path model_dir;
    path output_dir;

    app.add_option(
            "--model_dir",
            model_dir,
            "Directory where model is to be loaded from. Must contain 'actor.pt'."
            )->required();

    app.add_option(
            "--output_dir",
            output_dir,
            "Where to save results. Must not exist yet, will be created."
            )->required();

    try{
        app.parse(argc, argv);
    }
    catch (const CLI::ParseError &e) {
        return app.exit(e);
    }

    CPPTRACE_TRY {
        demo(model_dir, output_dir);
    } CPPTRACE_CATCH(const std::exception& e) {
        std::cerr<<"Exception: "<<e.what()<<std::endl;
        cpptrace::from_current_exception().print_with_snippets();
        throw e;
    }

    return 0;
}
