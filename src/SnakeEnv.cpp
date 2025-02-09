#include "SnakeEnv.hpp"
// #include "misc.hpp"
#include <SDL2/SDL.h>

#include <iostream>
#include <stdexcept>
#include <random>
#include <vector>

const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;

using std::runtime_error;
using std::cerr;
using std::random_device;
using std::vector;

namespace JungleGym{

SnakeEnv::SnakeEnv(int64_t width, int64_t height):
    observation_space(torch::zeros({width, height}, torch::kInt32)),
    action_space(torch::zeros({4}, torch::kInt32)),
    generator(random_device()()),
    width(width),
    height(height)
{
    if (width < 2 or height < 2) {
        throw std::runtime_error("ERROR: Cannot initialize 3-length snake in grid size < 2x2");
    }

    initialize_snake();
}


at::Tensor SnakeEnv::get_action_space() const {
    return action_space;
}


const at::Tensor& SnakeEnv::get_observation_space() const {
    return observation_space;
}


bool SnakeEnv::is_valid(const coord_t& coord) const {
    return coord.first >= 0 and coord.second >= 0 and coord.first < width and coord.second < height;
}


bool SnakeEnv::is_open(const coord_t& coord) const {
    return observation_space_2d[coord.first][coord.second] == 0;
}


void SnakeEnv::update_coord(const at::Tensor& action, coord_t& coord) const{
    auto a = action.argmax(0).item<int64_t>();

    switch (a) {
        case 0:
            coord.second += 1;   // UP
            break;
        case 1:
            coord.first += 1;    // RIGHT
            break;
        case 2:
            coord.second -= 1;   // DOWN
            break;
        case 3:
            coord.first -= 1;    // LEFT
            break;
        default:
            throw std::runtime_error("ERROR: Invalid range");
    }
}


void SnakeEnv::get_complement(at::Tensor& action) const{
    auto a = action.argmax(0).item<int64_t>();
    auto a_1d = action.accessor<int64_t,1>();
    action *= 0;
    
    switch (a) {
        case 0:
            a_1d[2] = 1;   // UP --> DOWN
            break;
        case 1:
            a_1d[3] = 1;   // RIGHT --> LEFT
            break;
        case 2:
            a_1d[0] = 1;   // DOWN --> UP
            break;
        case 3:
            a_1d[1] = 1;   // LEFT --> RIGHT
            break;
        default:
            throw std::runtime_error("ERROR: Invalid range");
    }
}


int64_t SnakeEnv::get_complement(int64_t a) const{
    switch (a) {
        case 0:
            return 2;   // UP --> DOWN
        case 1:
            return 3;   // RIGHT --> LEFT
        case 2:
            return 0;   // DOWN --> UP
        case 3:
            return 1;   // LEFT --> RIGHT
        default:
            throw std::runtime_error("ERROR: Invalid range");
    }
}


// Definitions for the virtual functions
void SnakeEnv::step(at::Tensor& action) {
    if (terminated or truncated){
        return;
    }

    std::unique_lock lock(m);  // Exclusive lock for writing

    // Even if the move is invalid we add it to the snake, and then test afterward
    snake.push_front(snake.front());

    update_coord(action, snake.front());

    if (not is_valid(snake.front())) {
        snake.pop_front();
        truncated = true;
        return;
    }

    if (not is_open(snake.front())) {
        snake.pop_front();
        truncated = true;
        return;
    }

    auto& tail = snake.back();

    // Update the grid (add head square)
    observation_space_2d[snake.front().first][snake.front().second] = 1;

    // Update the grid (remove tail square)
    observation_space_2d[tail.first][tail.second] = 0;

    snake.pop_back();
}


void SnakeEnv::reset() {
    observation_space *= 0;
    generator = mt19937(random_device()());
    initialize_snake();
}


void SnakeEnv::get_head(coord_t& coord) const {
    coord = snake.front();
}


void SnakeEnv::get_neck(coord_t& coord) const {
    coord = *(snake.begin()++);
}


void SnakeEnv::initialize_snake() {
    snake = {};

    auto action = get_action_space();
    auto action_1d = action.accessor<int32_t,1>();

    std::unique_lock lock(m);  // Exclusive lock for writing

    // Guaranteed valid starting point
    std::uniform_int_distribution<int64_t> x_dist(0, width-1);
    std::uniform_int_distribution<int64_t> y_dist(0, height-1);
    snake.emplace_front(x_dist(generator), y_dist(generator));
    observation_space_2d[snake.front().first][snake.front().second] = 1;
    vector<size_t> moves = {0,1,2,3};

    cerr << observation_space << '\n';

    while (snake.size() < 3) {
        snake.push_front(snake.front());

        std::shuffle(moves.begin(), moves.end(), generator);
        cerr << "-- head " << snake.front() << '\n';

        for (size_t i=0; i<=moves.size(); i++) {
            if (i == 4) {
                throw std::runtime_error("ERROR: cannot initialize 3-length snake");
            }

            auto a = moves[i];

            action *= 0;
            action_1d[a] = 1;

            update_coord(action, snake.front());

            cerr << "move " << a << '\n';
            cerr << "head " << snake.front() << '\n';

            if (is_valid(snake.front()) and is_open(snake.front())) {
                observation_space_2d[snake.front().first][snake.front().second] = 1;
                cerr << observation_space << '\n';
                break;
            }
            else {
                cerr << "fail" << '\n';
                cerr << is_valid(snake.front()) << ',' << is_open(snake.front()) << '\n';
                // Reset to prev position
                snake.front() = *(++snake.begin());
                cerr << "reset " << snake.front() << '\n';
            }
        }
    }

    // unique lock expires
}


void SnakeEnv::render() {
    int32_t SCREEN_WIDTH = 500;
    int32_t SCREEN_HEIGHT = SCREEN_WIDTH;
    int32_t w = SCREEN_WIDTH / (observation_space.sizes()[0] + 1);

    std::shared_lock lock(m);

    if (observation_space.sizes()[0] > SCREEN_WIDTH) {
        cerr << "WARNING: cannot render with more units than display size: " << observation_space.sizes()[0] <<  ',' << SCREEN_WIDTH << '\n';
        return;
    }

    lock.unlock();

    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return;
    }

    // Create a window
    SDL_Window* window = SDL_CreateWindow("SDL Squares", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                          SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    if (!window) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return;
    }

    // Create a renderer
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return;
    }

    // Main loop flag
    bool quit = false;
    SDL_Event e;

    // Main loop
    while (!quit) {
        // Handle events
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
        }

        // Clear screen with white
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); // White
        SDL_RenderClear(renderer);

        // Draw squares with different colors

        lock.lock();
        auto nz = observation_space.nonzero();
        auto accessor = nz.accessor<int64_t,2>();

        for (int32_t i=0; i < nz.size(0); i++) {
            auto x = w*int32_t(accessor[i][0]) + w/2;
            auto y = w*int32_t(accessor[i][1]) + w/2;

            SDL_Rect r({x+1,y+1,w-2,w-2});
            SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); // Red
            SDL_RenderFillRect(renderer, &r);
        }
        lock.unlock();

        // Present renderer to update the window
        SDL_RenderPresent(renderer);

        // Delay to control the frame rate (optional)
        cerr << "waiting" << '\n';
        SDL_Delay(16);
    }

    // Clean up and quit SDL
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}


void SnakeEnv::close() {
    // Implement close logic
}


}
