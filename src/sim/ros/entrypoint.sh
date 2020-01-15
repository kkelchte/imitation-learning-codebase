export PYTHONPATH=''
source /opt/ros/kinetic/setup.bash
export PYTHONPATH=/usr/lib/python2.7/dist-packages:$PYTHONPATH

# potentially perform catkin make if source files have changed.
cd "${HOME}/src/sim/ros" || exit 1
make catkin_ws/devel/setup.bash

# shellcheck disable=SC1090
source "${HOME}/src/sim/ros/catkin_ws/devel/setup.bash" --extend || exit 2
