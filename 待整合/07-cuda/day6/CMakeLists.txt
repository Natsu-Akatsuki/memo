cmake_minimum_required(VERSION 3.17)
project(day6 CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(day6 main.cu)

set_target_properties(
        day6
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)