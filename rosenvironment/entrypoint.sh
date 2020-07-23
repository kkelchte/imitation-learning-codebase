#!/usr/bin/env bash

export HOME="${PWD}"
export PYTHONPATH=''
source /opt/ros/melodic/setup.bash

# perform catkin make if source files have changed.
cd "${HOME}/src/sim/ros" || exit 1

if [ ! -d python3_ros_ws ] ; then
  make install_python3_ros_ws
fi
if [ ! -d python2_ros_ws ] ; then
  make install_python2_ros_ws
fi

make python2_ros_ws/devel/setup.bash
# shellcheck disable=SC1090
source "${HOME}/src/sim/ros/python2_ros_ws/devel/setup.bash" --extend || exit 2

make python3_ros_ws/devel/setup.bash
# shellcheck disable=SC1090
source "${HOME}/src/sim/ros/python3_ros_ws/devel/setup.bash" --extend || exit 2


export GAZEBO_MODEL_PATH="${HOME}/src/sim/ros/gazebo/models"
export PYTHONPATH=${PYTHONPATH}:${HOME}

export LD_LIBRARY_PATH=/opt/ros/melodic/lib:/home/klaas/code/imitation-learning-codebase/src/sim/ros/python2_ros_ws/devel/lib:/home/klaas/code/imitation-learning-codebase/src/sim/ros/python3_ros_ws/devel/lib:/.singularity.d/libs
export CMAKE_PREFIX_PATH=/opt/ros/melodic:/home/klaas/code/imitation-learning-codebase/src/sim/ros/python2_ros_ws/devel:/home/klaas/code/imitation-learning-codebase/src/sim/ros/python3_ros_ws/devel

cd "${HOME}" || exit 1

# potentially remove .singularity.d/libs from LD_LIBRARY_PATH
"$@"
