cmake_minimum_required(VERSION 3.22)
project(jungle_gym VERSION 0.0.0)

message(STATUS
    "Use the following flags for special configuration options E.g.: cmake .. -D[flag]=1 -D[flag2]=1
    debug:          invoke less optimization and more transparency for debugging (including ASan)
    no_san:         skip AddressSanitizer and other sanitizers (only relevant when using debug=1)
    dev:            skip fetching external projects (must NOT be invoked on first compilation)
    "
)

message(STATUS "CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "CMAKE_SYSTEM: ${CMAKE_SYSTEM}")

# Compiler options.
set(DOWNLOAD_EXTRACT_TIMESTAMP 1)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


if (debug)
    message(STATUS "---- Building with debug options ----")

    if (NOT no_san OR NOT DEFINED no_san)
        # Memory debug
        set(CMAKE_CXX_FLAGS "-fexceptions -fsanitize=thread -fPIE -pie")
        #        set(CMAKE_CXX_FLAGS "-fexceptions -fsanitize=address -fsanitize=undefined")
        #        set(ASAN_OPTIONS=check_initialization_order=1)
        set(ASAN_OPTIONS=detect_leaks=1)
    else()
        set(CMAKE_CXX_FLAGS "-fexceptions")
    endif()

    add_definitions(-ggdb3 -Og -Wall)       # Debugging + minimal optimization

else()
    message(STATUS "---- Building with optimization ----")

    # Standard compilation
    add_definitions(-g -O3 -Wall)              # Much optimization, but retain debug symbols
endif()


# Include header files
include_directories(
        "inc"
)

# Define our shared library sources. NOT test/executables.
set(SOURCES
        src/Environment.cpp
        src/Episode.cpp
        src/SnakeEnv.cpp
)

project(jungle_gym)
add_library(jungle_gym STATIC ${SOURCES})

# To make sure the library is named Name.so,
# get rid of the "lib" prefix.
set_target_properties(jungle_gym PROPERTIES PREFIX "")

# Eliminate an extraneous -D during compilation.
set_target_properties(jungle_gym PROPERTIES DEFINE_SYMBOL "")

# Need to explicitly enable ExternalProject functionality
include(ExternalProject)


# -------- DEPS --------

set(CMAKE_PREFIX_PATH "/home/ryan/software/libtorch/")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_include_directories(jungle_gym PUBLIC ${TORCH_INCLUDE_DIRS})

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

find_package(SDL2 REQUIRED)

##############################################
# ------------------------------------------ #
# -------- LINKING EXTERNAL LIBRARY -------- #
# ---------------- cpptrace ---------------- #
# ------------------------------------------ #
##############################################

include(ExternalProject)

message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

include(FetchContent)

FetchContent_Declare(
        cpptrace
        GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
        GIT_TAG        v0.7.5
)
FetchContent_MakeAvailable(cpptrace)

include_directories(
        ${cpptrace_SOURCE_DIR}/include
#        ${libdwarf_SOURCE_DIR}
#        ${libunwind_SOURCE_DIR}
)

message(STATUS "cpptrace location: ${cpptrace_SOURCE_DIR}")


# -------- TESTS --------

set(TESTS
    test_snake_env
    test_sdl2
)

foreach(FILENAME_PREFIX ${TESTS})
    add_executable(${FILENAME_PREFIX} src/test/${FILENAME_PREFIX}.cpp)

    target_link_libraries(${FILENAME_PREFIX}
            jungle_gym
            Threads::Threads
            "${TORCH_LIBRARIES}"
            SDL2::SDL2
            cpptrace::cpptrace
    )

endforeach()


# -------- EXECUTABLES --------

set(EXECUTABLES
        train_snake
)

foreach(FILENAME_PREFIX ${EXECUTABLES})
    add_executable(${FILENAME_PREFIX} src/executable/${FILENAME_PREFIX}.cpp)

    target_link_libraries(${FILENAME_PREFIX}
            jungle_gym
            Threads::Threads
            "${TORCH_LIBRARIES}"
            SDL2::SDL2
            cpptrace::cpptrace
    )

endforeach()


# -------- final steps --------

# Where to install
set(BINARY_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)

## The shared library goes to the bin directory.
#install(TARGETS
#        jungle_gym
#        jungle_gym
#        DESTINATION ${BINARY_INSTALL_DIR})

# Some potentially helpful messages
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
