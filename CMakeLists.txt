cmake_minimum_required(VERSION 3.16)

project(TinyTensorA64)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(
    tinytensor
    src/main.cpp
    src/tensor.cpp
)

target_include_directories(
    tinytensor
    PRIVATE include
)