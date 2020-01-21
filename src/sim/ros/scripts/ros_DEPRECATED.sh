#!/bin/bash

# shellcheck disable=SC1090
#source "${HOME}"/src/sim/ros/entrypoint.sh
export PYTHONPATH=/opt/ros/melodic/lib/python2.7/dist-packages
xvfb-run -a roslaunch "$@"