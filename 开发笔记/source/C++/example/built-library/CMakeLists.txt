cmake_minimum_required(VERSION 3.11)
project(tutorial)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

add_executable(signal_tutorial signal_tutorial.cpp)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(signal_tutorial ${OpenCV_LIBRARIES})


add_executable(t1 t1.cpp)
target_link_libraries(t1 ${OpenCV_LIBRARIES})

add_executable(reinterpret_cast_example reinterpret_cast_example.cpp)