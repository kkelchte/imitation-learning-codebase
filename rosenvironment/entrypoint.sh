#!/usr/bin/env bash

export HOME="${PWD}"
export PYTHONPATH=''
source /opt/ros/melodic/setup.bash

# perform catkin make if source files have changed.
cd "${HOME}/src/sim/ros" || exit 1

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
source "${HOME}/src/sim/ros/python2_ros_ws/devel/setup.bash" --extend || exit 2

# shellcheck disable=SC1090
source "${HOME}/src/sim/ros/python3_ros_ws/devel/setup.bash" --extend || exit 2


export GAZEBO_MODEL_PATH="${HOME}/src/sim/ros/gazebo/models"
export PYTHONPATH=${PYTHONPATH}:${HOME}

export LD_LIBRARY_PATH=/opt/ros/melodic/lib:${HOME}/src/sim/ros/python2_ros_ws/devel/lib:${HOME}/src/sim/ros/python3_ros_ws/devel/lib:/.singularity.d/libs
export CMAKE_PREFIX_PATH=/opt/ros/melodic:${HOME}/src/sim/ros/python2_ros_ws/devel:${HOME}/src/sim/ros/python3_ros_ws/devel

export DSO_PATH=${HOME}/src/sim/ros/dso
export TURTLEBOT3_MODEL='burger'
cd "${HOME}" || exit 1

# potentially remove .singularity.d/libs from LD_LIBRARY_PATH
"$@"
