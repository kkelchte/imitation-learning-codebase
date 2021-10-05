cmake_minimum_required(VERSION 2.8.3)

project(imitation_learning_ros_package)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    gazebo_ros
    cv_bridge
    sensor_msgs
    geometry_msgs
    nav_msgs
    image_transport
    message_generation
)
find_package(Boost REQUIRED COMPONENTS system signals thread)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS}
                           ${catkin_INCLUDE_DIRS}
                           ${OpenCV_INCLUDE_DIRS}
)
add_message_files(
    FILES
    RosReward.msg
    CombinedGlobalPoses.msg
)
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
)

generate_dynamic_reconfigure_options(
  cfg/pid.cfg
)

catkin_package(
    CATKIN_DEPENDS message_runtime
)
include_directories(
  rosnodes
)

catkin_install_python(PROGRAMS
  rosnodes/control_mapping.py
  rosnodes/fsm.py
  rosnodes/robot_mapping.py
  rosnodes/ros_expert.py
  rosnodes/mathias_controller.py
  rosnodes/altitude_control.py
  rosnodes/teleop_twist_keyboard.py
  rosnodes/waypoint_indicator.py
  rosnodes/modified_state_publisher.py
  rosnodes/modified_state_frame_visualizer.py
  rosnodes/april_tag_detector.py
  rosnodes/fgbg_actor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  PATTERN ".svn" EXCLUDE)


# make sure configure headers are built before any node using them
#add_dependencies(mathias_controller ${PROJECT_NAME}_gencfg)

