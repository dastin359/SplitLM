cmake_minimum_required(VERSION 3.21)
project(trilane CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama.cpp installed package (Snapdragon preset: CPU + OpenCL + Hexagon)
find_package(llama REQUIRED)

add_executable(trilane-split src/split_bench.cpp)
target_link_libraries(trilane-split PRIVATE llama)
