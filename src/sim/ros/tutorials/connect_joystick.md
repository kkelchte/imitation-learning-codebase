For some reason the joy_node is not found from the load_ros.launch file.
Therefore the joy_stick driver, named joy_node should be launched from a freshly sourced enviroment.
- Open a new terminal: CTR+ALT+T

´´´bash
cd rosenvironment/singularity
make singularity-shell
source /opt/ros/melodic/setup.bash
export ROS_NAMESPACE=/actor/joystick
rosrun joy joy_node _dev=/dev/input/js0
´´´
