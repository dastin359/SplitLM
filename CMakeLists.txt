cmake_minimum_required(VERSION 3.21)
project(trilane CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama.cpp installed package (Snapdragon preset: CPU + OpenCL + Hexagon)
find_package(llama REQUIRED)

add_executable(trilane-split src/split_bench.cpp)
target_link_libraries(trilane-split PRIVATE llama)

# OpenAI-compatible server + dashboard (vendored cpp-httplib / nlohmann json from llama.cpp)
set(LLAMA_SRC ${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp CACHE PATH "llama.cpp checkout (for vendored headers)")
set(LLAMA_VENDOR ${LLAMA_SRC}/vendor)
add_library(trilane-httplib STATIC ${LLAMA_VENDOR}/cpp-httplib/httplib.cpp)
target_include_directories(trilane-httplib PUBLIC ${LLAMA_VENDOR}/cpp-httplib)
target_compile_definitions(trilane-httplib PUBLIC CPPHTTPLIB_TCP_NODELAY=1 CPPHTTPLIB_LISTEN_BACKLOG=128)
target_compile_options(trilane-httplib PRIVATE -w)
target_link_libraries(trilane-httplib PUBLIC ws2_32)

add_executable(trilane-server src/server.cpp src/engine.cpp)
target_include_directories(trilane-server PRIVATE ${LLAMA_VENDOR})
target_link_libraries(trilane-server PRIVATE llama trilane-httplib)

# E1: CPU read bandwidth of HTP0 rpcmem vs heap
add_executable(trilane-rpcmem-bw src/rpcmem_bw.cpp)
target_link_libraries(trilane-rpcmem-bw PRIVATE llama)

# throttle diagnosis: memory bandwidth vs CPU compute
add_executable(trilane-probe src/probe.cpp)
target_link_libraries(trilane-probe PRIVATE pdh)
