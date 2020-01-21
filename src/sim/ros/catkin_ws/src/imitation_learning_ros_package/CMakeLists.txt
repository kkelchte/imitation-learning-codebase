cmake_minimum_required(VERSION 2.8.3)

project(imitation_learning_ros_package)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    gazebo_ros
    cv_bridge
    image_transport
)
find_package(Boost REQUIRED COMPONENTS system signals thread)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS}
                           ${catkin_INCLUDE_DIRS}
                           ${OpenCV_INCLUDE_DIRS}
)
catkin_package()

include_directories(
  rosnodes
)

catkin_install_python(PROGRAMS
  rosnodes/teleop_twist_keyboard.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE)
