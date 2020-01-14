#!/bin/bash

# shellcheck disable=SC1090
source "${HOME}"/src/sim/ros/entrypoint.sh

LAUNCHFILE="$1"
ARGUMENTS=${*:2}

#echo "LAUNCHFILE: ${LAUNCHFILE}"
#echo "ARGUMENTS: ${ARGUMENTS}"
echo "roslaunch ${LAUNCHFILE} ${ARGUMENTS}"
roslaunch "${LAUNCHFILE}" "${ARGUMENTS}"