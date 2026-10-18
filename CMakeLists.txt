cmake_minimum_required(VERSION 3.16)

project(TinyTensorA64)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(
    tinytensor_lib
    src/Tensor.cpp
    src/Linear.cpp
    src/Model.cpp
)

target_include_directories(
    tinytensor_lib
    PUBLIC include
)

add_executable(
    tinytensor
    src/main.cpp
)

target_link_libraries(
    tinytensor
    PRIVATE tinytensor_lib
)

add_executable(
    matmul_benchmark
    dot_benchmark
    benchmarks/benchmark_matmul.cpp
)

target_link_libraries(
    matmul_benchmark
    PRIVATE tinytensor_lib
)

enable_testing()

add_executable(
    tensor_tests
    tests/test_tensor.cpp
)

target_link_libraries(
    tensor_tests
    PRIVATE tinytensor_lib
)

add_test(
    NAME TensorTests
    COMMAND tensor_tests
)
