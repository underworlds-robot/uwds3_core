cmake_minimum_required(VERSION 2.8.3)
project(uwds3_core)

find_package(catkin REQUIRED COMPONENTS
  rospy
  uwds3_msgs
)

catkin_python_setup()

catkin_package(
  #INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS rospy uwds3_msgs
)
