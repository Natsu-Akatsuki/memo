cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example-app)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "~/anaconda3/envs/sleipnir/lib/python3.8/site-packages/torch/share/cmake")

find_package(Torch REQUIRED)
add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)