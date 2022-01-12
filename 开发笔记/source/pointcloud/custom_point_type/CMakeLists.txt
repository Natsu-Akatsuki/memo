cmake_minimum_required(VERSION 3.3)

project(custom_point_type)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

add_executable (custom_point_type custom_point_type.cpp)
target_link_libraries (custom_point_type ${PCL_LIBRARIES})
set_target_properties(custom_point_type PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})