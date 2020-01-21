export HOME="${PWD}"
export PYTHONPATH=''
source /opt/ros/melodic/setup.bash

# perform catkin make if source files have changed.
cd "${HOME}/src/sim/ros" || exit 1
make catkin_ws/devel/setup.bash
make extra_ros_ws/devel/setup.bash
cd "${HOME}" || exit 1

# shellcheck disable=SC1090
source "${HOME}/src/sim/ros/catkin_ws/devel/setup.bash" --extend || exit 2
# shellcheck disable=SC1090
source "${HOME}/src/sim/ros/extra_ros_ws/devel/setup.bash" --extend || exit 2

export GAZEBO_MODEL_PATH="${HOME}/src/sim/ros/gazebo/models"
export PYTHONPATH=${PYTHONPATH}:${PWD}

# potentially remove .singularity.d/libs from LD_LIBRARY_PATH
"$@"
