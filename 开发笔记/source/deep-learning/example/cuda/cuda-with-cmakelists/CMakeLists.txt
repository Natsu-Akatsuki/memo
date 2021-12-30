cmake_minimum_required(VERSION 3.11)
project(cuda-with-cmakelists)
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 14)

cuda_add_executable(cuda-with-cmakelists main.cu)

set_target_properties(cuda-with-cmakelists PROPERTIES CUDA_SEPARABLE_COMPILATION
  ON)

