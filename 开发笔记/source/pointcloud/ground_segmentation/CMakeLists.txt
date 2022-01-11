cmake_minimum_required(VERSION 3.5)
project(ground_segmentation)

add_compile_options(-Wno-dev)
set(CMAKE_CXX_STANDARD 14)


find_package(PCL REQUIRED QUIET)
include_directories(${PCL_INCLUDE_DIRS})

add_executable(ground_segmentation ground_segmentation.cpp)
target_link_libraries(ground_segmentation ${PCL_LIBRARIES})
set_target_properties(ground_segmentation PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}) 