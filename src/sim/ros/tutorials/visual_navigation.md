Learn blocks and meaning of parameters in ROS navigation stack from arxiv.org/pdf/1706.09068.pdf
Requirements:
git https://github.com/ros-planning/navigation.git (branch: melodic-devel)

Extra resources:
https://github.com/kkelchte/ros_course_part2
http://wiki.ros.org/navigation/Tutorials/Navigation%20Tuning%20Guide
http://wiki.ros.org/navigation/Tutorials/Writing%20A%20Global%20Path%20Planner%20As%20Plugin%20in%20ROS

Demo turtlebot gazebo:
roslaunch turtlebot3_gazebo turtlebot3_house.launch
roslaunch turtlebot3_navigation turtlebot3_navigation.launch map_file:=$HOME/src/sim/ros/python3_ros_ws/src/ros_course_part2/tb3_house_map/tb3_house_map.yaml

- Set estimated start location
- Set goal location

Demo turtlebot real:

 
