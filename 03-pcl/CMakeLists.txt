cmake_minimum_required(VERSION 3.10)

project(pcd_write)
set(CMAKE_BUILD_TYPE "Release")
find_package(PCL 1.8 REQUIRED QUIET)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (pcd_write pcd_write.cpp)
target_link_libraries (pcd_write ${PCL_LIBRARIES})

add_executable (pcd_read pcd_read.cpp)
target_link_libraries (pcd_read ${PCL_LIBRARIES})

#add_executable (planar_segmentation planar_segmentation.cpp)
#target_link_libraries (planar_segmentation ${PCL_LIBRARIES})
#
#add_executable (region_growing_segmentation region_growing_segmentation.cpp)
#target_link_libraries (region_growing_segmentation ${PCL_LIBRARIES})