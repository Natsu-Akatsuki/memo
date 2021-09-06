cmake_minimum_required(VERSION 3.17)
project(day3 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(day3 main.cu)

set_target_properties(
        day3
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)