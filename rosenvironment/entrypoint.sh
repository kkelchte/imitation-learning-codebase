export HOME=$PWD
export PYTHONPATH=''
source /opt/ros/melodic/setup.bash

# perform catkin make if source files have changed.
cd rosenvironment || exit 1
make catkin_ws/devel/setup.bash
cd ${HOME}

# shellcheck disable=SC1090
source "${HOME}/rosenvironment/catkin_ws/devel/setup.bash" --extend || exit 2

export GAZEBO_MODEL_PATH="${HOME}/src/sim/ros/gazebo/models"
export PYTHONPATH=${PYTHONPATH}:${PWD}

# potentially remove .singularity.d/libs from LD_LIBRARY_PATH

roscore &
"$@"
