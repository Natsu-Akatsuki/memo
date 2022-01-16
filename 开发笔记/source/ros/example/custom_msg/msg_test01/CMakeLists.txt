cmake_minimum_required(VERSION 3.0.2)
project(msg_test01)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
)

add_message_files(
  FILES
  test01.msg
)

generate_messages(
)

catkin_package(
  CATKIN_DEPENDS message_runtime rospy
)