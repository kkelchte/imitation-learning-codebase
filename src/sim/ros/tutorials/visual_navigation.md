Learn blocks and meaning of parameters in ROS navigation stack from arxiv.org/pdf/1706.09068.pdf
Requirements:
git https://github.com/ros-planning/navigation.git (branch: melodic-devel)
git https://github.com/ros-perception/openslam_gmapping.git (branch: melodic-devel)
git https://github.com/ros-perception/slam_gmapping.git (branch: melodic-devel)


Extra resources:
https://github.com/kkelchte/ros_course_part2
http://wiki.ros.org/navigation/Tutorials/Navigation%20Tuning%20Guide
http://wiki.ros.org/navigation/Tutorials/Writing%20A%20Global%20Path%20Planner%20As%20Plugin%20in%20ROS

Demo turtlebot gazebo:
0) start simulator
$ roslaunch turtlebot3_gazebo turtlebot3_house.launch

1) create map
$ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch 
$ roslaunch turtlebot3_slam turtlebot3_slam.launch slam_methods:=gmapping
# if you have turtlebot 3 also locally installed:
$ roslaunch $HOME/src/sim/ros/python3_ros_ws/src/turtlebot3/turtlebot3_teleop/launch/turtlebot3_teleop_key.launch
$ roslaunch $HOME/src/sim/ros/python3_ros_ws/src/turtlebot3/turtlebot3_slam/launch/turtlebot3_slam.launch slam_methods:=gmapping


2) drive in map
$ roslaunch $HOME/src/sim/ros/python3_ros_ws/src/turtlebot3/turtlebot3_navigation/launch/turtlebot3_navigation.launch map_file:=$HOME/src/sim/ros/python3_ros_ws/src/ros_course_part2/tb3_house_map/tb3_house_map.yaml

- Set estimated start location
- Set goal location

Demo turtlebot real:
0) start turtlebot, connect wifi to 'hotspot' (pw: turtlebot), login over ssh
(local)$ turtle
(local) $ roscore &
(local)$ ssh turtlebot@10.42.0.1 (pw: esat)
(remote)$ roslaunch turtlebot3_bringup turtlebot3_robot.launch

# (local)$ roslaunch turtlebot3_bringup turtlebot3_remote.launch

1) create map
# load rosmaster URI according to set IP addres (macbook: 177)
(local) $ turtle  
(local) $ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch &
(local) $ roslaunch turtlebot3_slam turtlebot3_slam.launch slam_methods:=gmapping
roslaunch &
(local) $ rosrun map_server map_saver -f $HOME/map

2) drive around in map
(local) $ turtle
(local) $ roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOME/map.yaml


 
