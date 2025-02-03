#include "SnakeEnv.hpp"
// #include "misc.hpp"

#include <iostream>
#include <stdexcept>
#include <random>

#include <SDL2/SDL.h>

const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;

using std::runtime_error;
using std::cerr;
using std::random_device;

using torch::zeros;

namespace JungleGym{

SnakeEnv::SnakeEnv(int64_t width, int64_t length):
    observation_space(torch::zeros({width, length}, torch::kInt32)),
    action_space(torch::zeros({4}, torch::kInt32)),
    generator(random_device()())
{}


SnakeEnv::SnakeEnv():
    observation_space(torch::zeros({32, 32}, torch::kInt32)),
    action_space(torch::zeros({4}, torch::kInt32)),
    generator(random_device()())
{}

// Definitions for the virtual functions
void SnakeEnv::step() {
    // Implement step logic
}

void SnakeEnv::reset() {
    // Implement reset logic
}

void SnakeEnv::render() {
    int32_t SCREEN_WIDTH = 500;
    int32_t SCREEN_HEIGHT = SCREEN_WIDTH;
    int32_t w = SCREEN_WIDTH / (observation_space.sizes()[0] + 1);

    if (observation_space.sizes()[0] > SCREEN_WIDTH) {
        cerr << "WARNING: cannot render with more units than display size: " << observation_space.sizes()[0] <<  ',' << SCREEN_WIDTH << '\n';
        return;
    }

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
