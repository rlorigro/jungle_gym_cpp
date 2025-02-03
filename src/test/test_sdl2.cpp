#include <iostream>
#include <random>
#include <stdexcept>

using std::runtime_error;
using std::cerr;

#include <SDL2/SDL.h>

const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;

int main(int argc, char* argv[]) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    // Create a window
    SDL_Window* window = SDL_CreateWindow("SDL Squares", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                          SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    if (!window) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    // Create a renderer
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    // Main loop flag
    bool quit = false;
    SDL_Event e;

    // Define a set of squares
    SDL_Rect squares[] = {
        { 100, 100, 50, 50 }, // Square 1
        { 200, 200, 60, 60 }, // Square 2
        { 400, 300, 40, 40 }  // Square 3
    };

    // Main loop
    while (!quit) {
        // Handle events
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
        }

        // Update square positions (e.g., move them to the right and down)
        for (int i = 0; i < 3; ++i) {
            squares[i].x += (rand() % 10) - 5; // Move square horizontally
            squares[i].y += (rand() % 10) - 5; // Move square vertically

            squares[i].x %= SCREEN_WIDTH; // Move square horizontally
            squares[i].y %= SCREEN_HEIGHT; // Move square vertically

            if (squares[i].x < 0){
                squares[i].x += SCREEN_WIDTH;
            }

            if (squares[i].y < 0){
                squares[i].y += SCREEN_HEIGHT;
            }
        }

        // Clear screen with white
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); // White
        SDL_RenderClear(renderer);

        // Draw squares with different colors
        SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); // Red
        SDL_RenderFillRect(renderer, &squares[0]);

        SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255); // Green
        SDL_RenderFillRect(renderer, &squares[1]);

        SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255); // Blue
        SDL_RenderFillRect(renderer, &squares[2]);

        // Present renderer to update the window
        SDL_RenderPresent(renderer);

        // Delay to control the frame rate (optional)
        SDL_Delay(16); // ~60 frames per second
    }

    // Clean up and quit SDL
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
