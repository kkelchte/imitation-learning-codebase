export PYTHONPATH=''
source /opt/ros/melodic/setup.bash
export PYTHONPATH=/usr/lib/python2.7/dist-packages:$PYTHONPATH

# perform catkin make if source files have changed.
cd "${HOME}/src/sim/ros" || exit 1
make catkin_ws/devel/setup.bash

# shellcheck disable=SC1090
source "${HOME}/src/sim/ros/catkin_ws/devel/setup.bash" --extend || exit 2

export GAZEBO_MODEL_PATH="${HOME}/src/sim/ros/gazebo/models"
#export PYTHONPATH=$PYTHONPATH:$HOME/tensorflow/pytorch_pilot TODO
# potentially remove .singularity.d/libs from LD_LIBRARY_PATH
