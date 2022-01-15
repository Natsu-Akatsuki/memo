cmake_minimum_required(VERSION 3.5)
project(basic_cplus_usage)

add_compile_options(-Wno-dev)
set(CMAKE_CXX_STANDARD 14)

find_package(PCL REQUIRED QUIET)
include_directories(${PCL_INCLUDE_DIRS})
add_executable(load_pointcloud load_pointcloud.cpp)
target_link_libraries(load_pointcloud ${PCL_LIBRARIES})
set_target_properties(load_pointcloud PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})