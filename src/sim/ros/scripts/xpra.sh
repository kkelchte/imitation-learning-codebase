#!/bin/bash

export HOME=$PWD

export XAUTHORITY=$HOME/.Xauthority
export DISPLAY=:$((100 + RANDOM % 154))

export LD_LIBRARY_PATH=''

XPRA_SYSTEMD_RUN=0 xpra --xvfb="Xorg -noreset -nolisten tcp
    -config ${HOME}/src/sim/ros/scripts/xorg.conf
    -logfile ${HOME}/.xpra/Xorg-${DISPLAY}.log" \
    start $DISPLAY
#    -config /etc/xpra/xorg.conf
#    > /dev/null 2>&1 &
