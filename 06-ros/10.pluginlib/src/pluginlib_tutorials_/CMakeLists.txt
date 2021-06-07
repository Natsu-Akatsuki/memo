cmake_minimum_required(VERSION 3.0.2)
project(pluginlib_tutorials_)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)


find_package(catkin REQUIRED COMPONENTS
  pluginlib
  roscpp
)



catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES pluginlib_tutorials_
#  CATKIN_DEPENDS pluginlib roscpp
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)



include_directories(include)
add_library(polygon_plugins src/polygon_plugins.cpp)
add_executable(polygon_loader src/polygon_loader.cpp)
target_link_libraries(polygon_loader ${catkin_LIBRARIES})