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
export PYTHONPATH=${PYTHONPATH}:${CODEDIR}

export LD_LIBRARY_PATH=/opt/ros/melodic/lib:${CODEDIR}/src/sim/ros/python2_ros_ws/devel/lib:${CODEDIR}/src/sim/ros/python3_ros_ws/devel/lib:/.singularity.d/libs
export CMAKE_PREFIX_PATH=/opt/ros/melodic:${CODEDIR}/src/sim/ros/python2_ros_ws/devel:${CODEDIR}/src/sim/ros/python3_ros_ws/devel

export DSO_PATH=${CODEDIR}/src/sim/ros/dso
export TURTLEBOT3_MODEL='burger'
cd "${CODEDIR}" || exit 1

#IPADDRESS="$(ip addr show wlp2s0 | grep inet | head -1 | cut -d '/' -f 1 | cut -d ' ' -f 6)"
#alias turtle='export ROS_MASTER_URI=http://${IPADDRESS}:11311 && export ROS_HOSTNAME={IPADDRESS}'
alias turtle='export ROS_MASTER_URI=http://192.168.0.149:11311 && export ROS_HOSTNAME=192.168.0.149'
# potentially remove .singularity.d/libs from LD_LIBRARY_PATH
"$@"
