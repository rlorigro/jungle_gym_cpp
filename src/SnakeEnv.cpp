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
using std::to_string;

namespace JungleGym{

SnakeEnv::SnakeEnv(int64_t width, int64_t height):
    observation_space(torch::zeros({width, height}, torch::kFloat32)),
    action_space(torch::zeros({4}, torch::kFloat32)),
    x_permutation(width),
    y_permutation(height),
    generator(random_device()()),
    width(width),
    height(height),
    i_x(0),
    i_y(0)
{
    if (width < 2 or height < 2) {
        throw std::runtime_error("ERROR: Cannot initialize 3-length snake in grid size < 2x2");
    }

    for (int64_t i = 0; i < width; i++) {
        x_permutation[i] = i;
    }

    for (int64_t i = 0; i < height; i++) {
        y_permutation[i] = i;
    }

    std::ranges::shuffle(x_permutation, generator);
    std::ranges::shuffle(y_permutation, generator);

    initialize_snake();
    add_apple(true);
}


torch::Tensor SnakeEnv::get_action_space() const {
    return action_space;
}


int64_t SnakeEnv::get_prev_action() const {
    return cached_action;
}


const torch::Tensor& SnakeEnv::get_observation_space() const {
    return observation_space;
}


bool SnakeEnv::is_valid(const coord_t& coord) const {
    return coord.first >= 0 and coord.second >= 0 and coord.first < width and coord.second < height;
}


bool SnakeEnv::is_open(const coord_t& coord) const {
    return int64_t(round(observation_space_2d[coord.first][coord.second])) != SNAKE_BODY;
}


void SnakeEnv::update_coord(int64_t a, coord_t& coord) const{
    switch (a) {
        case UP:
            coord.second += 1;   // UP
            break;
        case RIGHT:
            coord.first += 1;    // RIGHT
            break;
        case DOWN:
            coord.second -= 1;   // DOWN
            break;
        case LEFT:
            coord.first -= 1;    // LEFT
            break;
        default:
            throw std::runtime_error("ERROR: Invalid action range: " + to_string(a));
    }
}


void SnakeEnv::get_complement(torch::Tensor& action) const{
    auto a = action.argmax(0).item<int64_t>();
    auto a_1d = action.accessor<int64_t,1>();
    action *= 0;
    
    switch (a) {
        case UP:
            a_1d[DOWN] = 1;   // UP --> DOWN
            break;
        case RIGHT:
            a_1d[LEFT] = 1;   // RIGHT --> LEFT
            break;
        case DOWN:
            a_1d[UP] = 1;   // DOWN --> UP
            break;
        case LEFT:
            a_1d[RIGHT] = 1;   // LEFT --> RIGHT
            break;
        default:
            throw std::runtime_error("ERROR: Invalid range");
    }
}


int64_t SnakeEnv::get_complement(int64_t a) const{
    switch (a) {
        case UP:
            return DOWN;   // UP --> DOWN
        case RIGHT:
            return LEFT;   // RIGHT --> LEFT
        case DOWN:
            return UP;   // DOWN --> UP
        case LEFT:
            return RIGHT;   // LEFT --> RIGHT
        default:
            throw std::runtime_error("ERROR: Invalid range");
    }
}


void SnakeEnv::add_apple_unsafe() {
    // The apple positions are predestined by the x/y_permutation vectors.
    for (size_t i=0; i<width; i++) {
        auto x = x_permutation[i_x%width];

        for (size_t j=0; j<width; j++) {
            auto y = y_permutation[i_y%width];

            if (int64_t(round(observation_space_2d[x][y])) != SNAKE_BODY) {
                observation_space_2d[x][y] = APPLE;
                apple = {x,y};
                return;
            }
            i_y++;
        }
        i_x++;
    }

    // This should be unreachable because it means all squares have been checked
    // Will deal with this error later (good problem to have)
    throw runtime_error("ERROR: cannot allocate apple: you win");
}


// Definitions for the virtual functions
void SnakeEnv::add_apple(bool use_lock) {
    if (use_lock) {
        std::unique_lock lock(m);
        add_apple_unsafe();
    }
    else {
        add_apple_unsafe();
    }
}


// Definitions for the virtual functions
void SnakeEnv::step() {
    step(cached_action);
}


// Definitions for the virtual functions
void SnakeEnv::step(const torch::Tensor& action) {
    auto a = int64_t(action.argmax(0).item<int64_t>());

    step(a);
}


// Definitions for the virtual functions
void SnakeEnv::step(int64_t a) {
    reward = 0;

    if (terminated or truncated){
        return;
    }

    cached_action = a;

    std::unique_lock lock(m);  // Exclusive lock for writing

    // Even if the move is invalid we add it to the snake, and then test afterward
    snake.push_front(snake.front());

    update_coord(a, snake.front());

    if (not is_valid(snake.front())) {
        snake.pop_front();
        truncated = true;
        reward = REWARD_COLLISION;
        return;
    }

    if (not is_open(snake.front())) {
        snake.pop_front();
        truncated = true;
        reward = REWARD_COLLISION;
        return;
    }

    auto& tail = snake.back();

    // Update the grid (add head square)
    observation_space_2d[snake.front().first][snake.front().second] = SNAKE_HEAD;

    coord_t neck;
    get_neck(neck);

    // Need to set the previous the square back down to BODY value
    observation_space_2d[neck.first][neck.second] = SNAKE_BODY;

    if (snake.front() == apple) {
        cerr << "mmm ... delicious" << '\n';
        add_apple(false);
        reward += REWARD_APPLE;
    }
    else {
        // Update the grid (remove tail square)
        // Only trim tail when not eatin a apple
        observation_space_2d[tail.first][tail.second] = 0;
        snake.pop_back();
        reward += REWARD_MOVE;
    }
}


void SnakeEnv::reset() {
    observation_space *= 0;
    generator = mt19937(random_device()());
    terminated = false;
    truncated = false;
    reward = 0;
    cached_action.store(0);
    initialize_snake();
    std::ranges::shuffle(x_permutation, generator);
    std::ranges::shuffle(y_permutation, generator);
    i_x = 0;
    i_y = 0;
    add_apple(true);
}


void SnakeEnv::get_head(coord_t& coord) const {
    coord = snake.front();
}


void SnakeEnv::get_neck(coord_t& coord) const {
    coord = *(snake.begin()++);
}


void SnakeEnv::initialize_snake() {
    snake = {};

    std::unique_lock lock(m);  // Exclusive lock for writing

    // Guaranteed valid starting point
    std::uniform_int_distribution<int64_t> x_dist(0, width-1);
    std::uniform_int_distribution<int64_t> y_dist(0, height-1);
    snake.emplace_front(x_dist(generator), y_dist(generator));
    observation_space_2d[snake.front().first][snake.front().second] = 1;
    vector<int64_t> moves = {0,1,2,3};

    while (snake.size() < 3) {
        snake.push_front(snake.front());

        std::shuffle(moves.begin(), moves.end(), generator);

        for (size_t i=0; i<=moves.size(); i++) {
            if (i == 4) {
                throw std::runtime_error("ERROR: cannot initialize 3-length snake");
            }

            auto a = moves[i];

            update_coord(a, snake.front());

            if (is_valid(snake.front()) and is_open(snake.front())) {
                observation_space_2d[snake.front().first][snake.front().second] = 1;

                // Keep track of prev action even when initializing so it can be known at start
                cached_action = a;
                break;
            }
            else {
                // Reset to prev position
                snake.front() = *(++snake.begin());
            }
        }
    }

    // unique lock expires
}


void SnakeEnv::render(bool interactive) {
    int32_t SCREEN_WIDTH = 500;
    int32_t SCREEN_HEIGHT = SCREEN_WIDTH;
    int32_t w = SCREEN_WIDTH / int32_t(observation_space.sizes()[0] + 1);

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

            // Handle other events, such as keyboard input
            if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_ESCAPE) {
                    quit = true;
                }
                else if (e.key.keysym.sym == SDLK_UP) {
                    cerr << "<-" << '\n';
                    cached_action = UP;
                }
                else if (e.key.keysym.sym == SDLK_RIGHT) {
                    cerr << "/\\" << '\n';
                    cached_action = RIGHT;
                }
                else if (e.key.keysym.sym == SDLK_DOWN) {
                    cerr << "->" << '\n';
                    cached_action = DOWN;
                }
                else if (e.key.keysym.sym == SDLK_LEFT) {
                    cerr << "\\/" << '\n';
                    cached_action = LEFT;
                }
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
            auto i_x = accessor[i][0];
            auto i_y = accessor[i][1];

            // Rendering on Y axis is inverted, so we un-invert y
            auto x = w*int32_t(i_x) + w/2;
            auto y = SCREEN_HEIGHT - (w*int32_t(i_y) + w/2) - w;

            SDL_Rect r({x+1,y+1,w-2,w-2});

            if (i_x == apple.first and i_y == apple.second) {
                SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255); // Red
            }
            else {
                SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); // Red
            }

            SDL_RenderFillRect(renderer, &r);
        }
        lock.unlock();

        // Present renderer to update the window
        SDL_RenderPresent(renderer);

        // Delay to control the frame rate (optional)
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
