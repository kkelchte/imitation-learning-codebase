## Background

Learn blocks and meaning of parameters in ROS navigation stack from arxiv.org/pdf/1706.09068.pdf

## Installation

Requirements:
```bash
git https://github.com/ros-planning/navigation.git (branch: melodic-devel)
git https://github.com/ros-perception/openslam_gmapping.git (branch: melodic-devel)
git https://github.com/ros-perception/slam_gmapping.git (branch: melodic-devel)
```

Extra resources:
```bash
https://github.com/kkelchte/ros_course_part2
http://wiki.ros.org/navigation/Tutorials/Navigation%20Tuning%20Guide
http://wiki.ros.org/navigation/Tutorials/Writing%20A%20Global%20Path%20Planner%20As%20Plugin%20in%20ROS
```

## Demo turtlebot gazebo:

### 0) start simulator
```bash
$ roslaunch turtlebot3_gazebo turtlebot3_house.launch
```

### 1) create map
```
$ roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch 
$ roslaunch turtlebot3_slam turtlebot3_slam.launch slam_methods:=gmapping
# if you have turtlebot 3 also locally installed:
$ roslaunch $CODEDIR/src/sim/ros/python3_ros_ws/src/turtlebot3/turtlebot3_teleop/launch/turtlebot3_teleop_key.launch
$ roslaunch $CODEDIR/src/sim/ros/python3_ros_ws/src/turtlebot3/turtlebot3_slam/launch/turtlebot3_slam.launch slam_methods:=gmapping
```

### 2) drive in map

```
$ roslaunch $CODEDIR/src/sim/ros/python3_ros_ws/src/turtlebot3/turtlebot3_navigation/launch/turtlebot3_navigation.launch map_file:=$HOME/src/sim/ros/python3_ros_ws/src/ros_course_part2/tb3_house_map/tb3_house_map.yaml
```

- Set estimated start location
- Set goal location

## Demo turtlebot real:

### 0) start turtlebot, connect wifi to 'hotspot' (pw: turtlebot), login over ssh
The IP Address provided corresponds to my home address. 
Depending on your WIFI setting access the correct WIFI address of both your host machine and the Turtlebot with: `$ ip addr`.
The following three environment variables need to be updated accordingly:

```
# on your host machine
ROS_MASTER_URI=http://HOST_IP_ADDRESS:11311 && export ROS_HOSTNAME=HOST_IP_ADDRESS
# on raspberry pi of turtlebot
ROS_MASTER_URI=http://HOST_IP_ADDRESS:11311 && export ROS_HOSTNAME=RPI_IP_ADDRESS
```

Easiest is to add an alias in the bashrc of the Turtlebot and 
add a 'turtle' alias in the entrypoint within your codebase.

```
(local)$ turtle
(local)$ roscore &
(local)$ ssh turtlebot@192.168.0.167 (pw: departmentsname)
(remote)$ roslaunch turtlebot3_bringup turtlebot3_robot.launch
```

### 1) load turtle environment, create map and save map.

```
# load rosmaster URI according to set IP addres (macbook: 177)
(local) $ turtle
(local) $ roslaunch imitation_learning_ros_package teleop_joystick.launch
(local) $ roslaunch $CODEDIR/src/sim/ros/python3_ros_ws/src/turtlebot3/turtlebot3_slam/launch/turtlebot3_slam.launch slam_methods:=gmapping &
(local) $ rosrun map_server map_saver -f $HOME/src/sim/ros/map
```


### 2) drive around in map
```
(local) $ turtle
(local) $ roslaunch $HOME/src/sim/ros/python3_ros_ws/src/turtlebot3/turtlebot3_navigation/launch/turtlebot3_navigation.launch map_file:=$HOME/src/sim/ros/map.yaml
```

### 3) navigate in living room and parse visual information

```
$ turtle
$ python3.8 src/sim/ros/python3_ros_ws/src/ros_course_part2/src/topic03_map_navigation/navigate_goal.py
```
### Troubleshoot:

__Cost map not loading, tf-updates are out of date (>10**8)__

In case the TF frames are not updated due to max time delay surpassed, the cost map will not load. 
This is probably due to the wrong time setting on the raspberry pi of the Turtlebot.
Normally the Turtlebot should automatically connect to a near Wifi spot and with this update the time.
However, if you start roslaunch too quickly it will not have reset the time yet and give this error.


 
