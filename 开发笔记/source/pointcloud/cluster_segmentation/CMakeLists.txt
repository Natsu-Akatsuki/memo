cmake_minimum_required(VERSION 3.3)

project(cluster_segmentation)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (cluster_segmentation cluster_segmentation.cpp)
target_link_libraries (cluster_segmentation ${PCL_LIBRARIES})
set_target_properties(cluster_segmentation PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}) 