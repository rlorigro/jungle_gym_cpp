#include "SnakeEnv.hpp"
// #include "misc.hpp"
#include <SDL2/SDL.h>

#include <iostream>
#include <stdexcept>
#include <random>
#include <vector>
#include <cmath>

using std::runtime_error;
using std::random_device;
using std::to_string;
using std::vector;
using std::atan2;
using std::cos;
using std::sin;
using std::min;
using std::cerr;

using torch::slice;
using namespace torch::indexing;

namespace JungleGym{

SnakeEnv::SnakeEnv(int64_t width, int64_t height):
    observation_space(torch::zeros({width, height, 4}, torch::kFloat32)),
    observation_space_3d(observation_space.accessor<float,3>()),
    action_space(torch::zeros({3}, torch::kFloat32)),
    generator(random_device()()),
    width(width),
    height(height),
    i_permutation(0)
{
    fill_wall();

    if (width < 3 or height < 3) {
        throw std::runtime_error("ERROR: Cannot initialize 3-length snake in grid size < 2x2");
    }

    for (int64_t i=1; i < width-1; i++) {
        for (int64_t j=1; j < height-1; j++) {
            xy_permutation.emplace_back(i, j);
        }
    }

    std::ranges::shuffle(xy_permutation, generator);

    initialize_snake();
    add_apple(true);
}


void SnakeEnv::reset() {
    // Exclusive lock for writing
    std::unique_lock lock(m);

    observation_space *= 0;
    fill_wall();
    generator = mt19937(random_device()());
    terminated = false;
    truncated = false;
    reward = 0;
    cached_action.store(STRAIGHT);
    initialize_snake();
    std::ranges::shuffle(xy_permutation, generator);
    i_permutation = 0;
    add_apple(false);
}


void SnakeEnv::fill_wall() {
    observation_space.index({0,"...", WALL}) = 1;
    observation_space.index({width-1,"...", WALL}) = 1;

    observation_space.index({"...",0, WALL}) = 1;
    observation_space.index({"...",height-1, WALL}) = 1;
}


void SnakeEnv::initialize_snake() {
    std::uniform_int_distribution<size_t> length_dist(initial_length_min,initial_length_max);

    // Guaranteed valid starting point
    std::uniform_int_distribution<int64_t> x_dist(1, width-2);
    std::uniform_int_distribution<int64_t> y_dist(1, height-2);

    snake = {};

    snake.emplace_front(x_dist(generator), y_dist(generator));

    observation_space_3d[snake.front().first][snake.front().second][SNAKE_BODY] = 1;

    vector<int64_t> moves = {0,1,2};

    while (snake.size() < length_dist(generator)) {
        coord_t next = snake.front();

        std::shuffle(moves.begin(), moves.end(), generator);

        for (size_t i=0; i<=moves.size(); i++) {
            if (i == moves.size() and snake.size() > 1) {
                // Rewind if failed to add anything to this snake
                observation_space_3d[snake.front().first][snake.front().second][SNAKE_BODY] = 0;
                snake.pop_front();
                continue;
            }

            auto a = moves[i];

            update_coord(a, next);

            if (is_open(next)) {
                observation_space_3d[next.first][next.second][SNAKE_BODY] = 1;
                snake.emplace_front(next);
                break;
            }
            else {
                // Reset to prev position
                next = snake.front();
            }
        }
    }

    observation_space_3d[snake.front().first][snake.front().second][SNAKE_HEAD] = 1;
    observation_space_3d[snake.front().first][snake.front().second][SNAKE_BODY] = 0;

    // Default to always going straight next time unless specified otherwise
    cached_action = STRAIGHT;

    patience_counter = 0;
}


torch::Tensor SnakeEnv::get_action_space() const {
    return action_space;
}


/**
 * Return a copy of the observation space that has been adjusted to make it "machine readable". I.e.,
 * the model needs to know the sequence of blocks that make the body, in order, so we add a "decay" in "brightness"
 * @return Modified copy of the observation space
 */
torch::Tensor SnakeEnv::get_observation_space() const {
    auto o = observation_space.clone();

    float a = 1.0;
    float b = 0.1;
    float delta = (a-b) / float(snake.size() - 2);

    size_t i = 0;
    for (auto [x,y]: snake) {
        if (i == 0) {
            // Don't rescale head pos
            i++;
            continue;
        }

        // cerr << x << ',' << y << ',' << SNAKE_BODY << '=' << a - float(i-1)*delta << ' ' << o.sizes() << '\n';
        o.index({x,y,SNAKE_BODY}) = a - float(i-1)*delta;
        i++;
    }

    // Mean = 0, Stddev = 1, Shape = etc
    torch::Tensor noise = torch::normal(0.0, 0.01, o.sizes());
    o += noise;
    o.clip_(0.0, 1.0);

    return o;
}


bool SnakeEnv::is_open(const coord_t& coord) const {
    bool not_body = (int64_t(round(observation_space_3d[coord.first][coord.second][SNAKE_BODY])) == 0);
    bool not_head = (int64_t(round(observation_space_3d[coord.first][coord.second][SNAKE_HEAD])) == 0);
    bool not_wall = (int64_t(round(observation_space_3d[coord.first][coord.second][WALL])) == 0);
    return not_body and not_head and not_wall;
}


void SnakeEnv::update_coord(int64_t a, coord_t& coord) const{
    float theta;

    if (snake.size() >= 2) {
        coord_t head;
        coord_t neck;

        get_head(head);
        get_neck(neck);

        float x_delta = float(head.first - neck.first);
        float y_delta = float(head.second - neck.second);

        if (x_delta + y_delta >= 1.01) {
            throw runtime_error("ERROR: illegal move in snake environment: " + to_string(head.first) + "," + to_string(head.second) + "-->" + to_string(neck.first) + "," + to_string(neck.second));
        }

        theta = atan2(y_delta,x_delta);
    }
    else {
        // std::uniform_int_distribution<int64_t> theta_dist(0,4);
        // theta = pi*float(theta_dist(generator))/2;
        theta = 0;
    }

    switch (a) {
        case LEFT:
            theta += pi/2;
            break;
        case STRAIGHT:
            break;
        case RIGHT:
            theta -= pi/2;
            break;
        default:
            throw std::runtime_error("ERROR: Invalid action range: " + to_string(a));
    }

    coord.first += round(cos(theta));
    coord.second += round(sin(theta));
}


void SnakeEnv::add_apple_unsafe() {
    // The apple positions are predestined by the xy_permutation vector. By iterating this vector we are guaranteed to
    // observe every coordinate, in random order
    for (size_t i=0; i<width; i++) {
        size_t p = i_permutation%xy_permutation.size();
        auto [x,y] = xy_permutation[p];

        if (is_open({x,y})) {
            observation_space_3d[x][y][APPLE] = 1;
            apple = {x,y};
            return;
        }

        i_permutation++;
    }

    // This should be unreachable because it means all squares have been checked
    // Otherwise the game has been "won"
    terminated = true;
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
void SnakeEnv::step(int64_t a) {
    std::unique_lock lock(m);  // Exclusive lock for writing

    reward = 0;
    cached_action = STRAIGHT;

    if (terminated or truncated){
        return;
    }

    coord_t next = snake.front();

    update_coord(a, next);

    if (not is_open(next)) {
        terminated = true;
        reward = REWARD_COLLISION;
        return;
    }

    snake.emplace_front(next);

    coord_t neck;
    get_neck(neck);

    // Update the grid (add head square)
    observation_space_3d[snake.front().first][snake.front().second][SNAKE_HEAD] = 1;

    // Need to remove head indicator from prev position
    observation_space_3d[neck.first][neck.second][SNAKE_HEAD] = 0;

    // The body trails the head
    observation_space_3d[neck.first][neck.second][SNAKE_BODY] = 1;

    auto& tail = snake.back();

    if (snake.front() == apple) {
        cerr << "mmm ... delicious" << '\n';
        observation_space_3d[apple.first][apple.second][APPLE] = 0;
        add_apple(false);
        reward += REWARD_APPLE;
        patience_counter = 0;
    }
    else {
        // Update the grid (remove tail square)
        // Only trim tail when not eatin a apple
        observation_space_3d[tail.first][tail.second][SNAKE_BODY] = 0;
        snake.pop_back();
        reward += REWARD_MOVE;
        patience_counter++;
    }

    if (patience_counter == patience_limit) {
        truncated = true;
    }

    // cerr << "step" << '\n';
    // cerr << observation_space.sum(2,false) << '\n';
}


void SnakeEnv::get_head(coord_t& coord) const {
    coord = snake.front();
}


void SnakeEnv::get_neck(coord_t& coord) const {
    if (snake.size() < 2) {
        throw std::runtime_error("ERROR: cannot get neck of snake with length: " + to_string(snake.size()));
    }

    coord = *(++snake.begin());
}


void SnakeEnv::render(bool interactive) {
    int32_t SCREEN_WIDTH = 600;
    int32_t SCREEN_HEIGHT = SCREEN_WIDTH;
    int32_t w = SCREEN_WIDTH / int32_t(observation_space.sizes()[0]);

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
                if (e.key.keysym.sym == SDLK_TAB) {
                    truncated = true;
                }
                else if (e.key.keysym.sym == SDLK_RIGHT) {
                    cerr << "-->" << '\n';
                    cached_action = RIGHT;
                }
                else if (e.key.keysym.sym == SDLK_LEFT) {
                    cerr << "<--" << '\n';
                    cached_action = LEFT;
                }
            }
        }

        // Clear screen with white
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); // White
        SDL_RenderClear(renderer);

        lock.lock();
        auto o = get_observation_space();
        auto o_3d = o.accessor<float,3>();
        lock.unlock();

        for (int64_t i_x=0; i_x<o.sizes()[0]; i_x++) {
            for (int64_t i_y=0; i_y<o.sizes()[1]; i_y++) {
                // Rendering on Y axis is inverted, so we un-invert y
                auto x = w*int32_t(i_x);
                auto y = SCREEN_HEIGHT - w*int32_t(i_y) - w;

                SDL_Rect r({x,y,w,w});

                const auto head = o_3d[i_x][i_y][SNAKE_HEAD];
                const auto body = o_3d[i_x][i_y][SNAKE_BODY];
                const auto apple = o_3d[i_x][i_y][APPLE];
                const auto wall = o_3d[i_x][i_y][WALL];

                SDL_SetRenderDrawColor(renderer, 200*body + 200*head, 200*apple + 200*head, 200*wall + 200*head, 200); // Red
                SDL_RenderFillRect(renderer, &r);
            }
        }

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

shared_ptr<Environment> SnakeEnv::clone() const{
    return make_shared<SnakeEnv>(width, height);
}


}
