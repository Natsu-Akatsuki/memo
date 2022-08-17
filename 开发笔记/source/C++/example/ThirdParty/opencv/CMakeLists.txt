cmake_minimum_required(VERSION 3.11)
project(contour)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(contour contour.cpp)
target_link_libraries(contour ${OpenCV_LIBS})

add_executable(segmentation segmentation.cpp)
target_link_libraries(segmentation ${OpenCV_LIBS})

add_executable(kalman kalman.cpp)
target_link_libraries(kalman ${OpenCV_LIBS})