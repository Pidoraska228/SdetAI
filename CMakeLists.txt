cmake_minimum_required(VERSION 3.20)
project(SdetAI LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    add_compile_options(/W4 /permissive- /utf-8 /Zc:__cplusplus /arch:AVX2)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    add_compile_options(-Wall -Wextra -O3)
endif()

include(FetchContent)
file(DOWNLOAD "https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp" ${CMAKE_BINARY_DIR}/json.hpp)
add_library(nlohmann_json INTERFACE IMPORTED)
set_target_properties(nlohmann_json PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR})

add_library(sdetai STATIC
    sparse_dynamic_nn.cpp
    low_rank.cpp
    quantization.cpp
    training.cpp
    bpe_tokenizer.cpp
)
target_link_libraries(sdetai PUBLIC nlohmann_json)

add_library(sdetai_http STATIC http_client.cpp)

add_executable(sdetai_demo demo_main.cpp)
target_link_libraries(sdetai_demo PRIVATE sdetai)

add_executable(sdetai_train train_main.cpp)
target_link_libraries(sdetai_train PRIVATE sdetai)

add_executable(sdetai_benchmark benchmark_test.cpp)
target_link_libraries(sdetai_benchmark PRIVATE sdetai)

add_executable(sdetai_chat sdetai_chat.cpp)
target_link_libraries(sdetai_chat PRIVATE sdetai)