#!/bin/bash

export HOME=$PWD
source /opt/ros/kinetic/setup.bash
# shellcheck disable=SC1090
source $HOME/src/sim/ros/catkin_ws/devel/setup.bash --extend

