#!/usr/bin/env bash

export CODEDIR="${PWD}"
export PYTHONPATH=''
export DATADIR="${CODEDIR}/experimental_data"
mkdir -p $DATADIR
source /opt/ros/melodic/setup.bash

# perform catkin make if source files have changed.
cd "${CODEDIR}/src/sim/ros" || exit 1

if [ ! -d python2_ros_ws ] ; then
  make install_python2_ros_ws
  make python2_ros_ws/devel/setup.bash
fi
if [ ! -d python3_ros_ws/devel ] ; then
  make install_python3_ros_ws
  make python3_ros_ws/devel/setup.bash
  rm -r python3_ros_ws/devel
  make python3_ros_ws/devel/setup.bash
fi

# shellcheck disable=SC1090
source "${CODEDIR}/src/sim/ros/python2_ros_ws/devel/setup.bash" --extend || exit 2

# shellcheck disable=SC1090
source "${CODEDIR}/src/sim/ros/python3_ros_ws/devel/setup.bash" --extend || exit 2


export GAZEBO_MODEL_PATH="${CODEDIR}/src/sim/ros/gazebo/models"
export GAZEBO_RESOURCE_PATH="/usr/share/gazebo-9:${CODEDIR}/src/sim/ros/gazebo"
export FGBG="$HOME/code/contrastive-learning"
export PYTHONPATH=${PYTHONPATH}:${CODEDIR}:${FGBG}

export LD_LIBRARY_PATH=/opt/ros/melodic/lib:${CODEDIR}/src/sim/ros/python2_ros_ws/devel/lib:${CODEDIR}/src/sim/ros/python3_ros_ws/devel/lib:/.singularity.d/libs
export CMAKE_PREFIX_PATH=/opt/ros/melodic:${CODEDIR}/src/sim/ros/python2_ros_ws/devel:${CODEDIR}/src/sim/ros/python3_ros_ws/devel

export DSO_PATH=${CODEDIR}/src/sim/ros/dso
export TURTLEBOT3_MODEL='burger'
cd "${CODEDIR}" || exit 1

HOST_IP_ADDRESS="$(ip addr | grep inet | grep 192 |  cut -d '/' -f 1 | cut -d ' ' -f 6)"
alias turtle='export ROS_MASTER_URI=http://${HOST_IP_ADDRESS}:11311 && export ROS_HOSTNAME=${HOST_IP_ADDRESS}'
alias bebop_demo='python3.8 src/sim/ros/src/online_evaluation_fgbg_real.py'
alias bebop_demo_full='export ROS_MASTER_URI=http://192.168.42.9:11311 && export ROS_HOSTNAME=192.168.42.9 && python3.8 src/sim/ros/src/online_evaluation_fgbg_real.py'
alias bebop_edge='export ROS_MASTER_URI=http://192.168.42.9:11311 && export ROS_HOSTNAME=192.168.42.72'

#alias turtle='export ROS_MASTER_URI=http://192.168.0.149:11311 && export ROS_HOSTNAME=192.168.0.149'
#alias turtle='export ROS_MASTER_URI=http://192.168.0.129:11311 && export ROS_HOSTNAME=192.168.0.129'
# potentially remove .singularity.d/libs from LD_LIBRARY_PATH
"$@"
