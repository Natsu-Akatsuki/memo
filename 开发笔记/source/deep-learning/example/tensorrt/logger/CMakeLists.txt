cmake_minimum_required(VERSION 3.0.2)
project(logger)

include_directories($ENV{HOME}/application/TensorRT-7.2.3.4/include)
add_executable(tensorrt_logger tensorrt_logger.cpp)
