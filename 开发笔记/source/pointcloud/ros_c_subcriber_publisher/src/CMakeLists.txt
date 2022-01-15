cmake_minimum_required(VERSION 3.3)
project(subscriber_publisher)

set(CMAKE_BUILD_TYPE "Release")

find_package(
  catkin REQUIRED
  COMPONENTS geometry_msgs
             sensor_msgs
             roscpp
             rospy
             std_msgs
             pcl_ros
             tf)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED QUIET)

include_directories(${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR}
                    ${PCL_INCLUDE_DIRS})

catkin_package(
  CATKIN_DEPENDS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  std_msgs
  DEPENDS
  EIGEN3
  PCL
  INCLUDE_DIRS)

add_executable(subscriber_publisher subscriber_publisher.cpp)
target_link_libraries(subscriber_publisher ${catkin_LIBRARIES} ${PCL_LIBRARIES})
