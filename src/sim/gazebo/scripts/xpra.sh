#!/bin/bash
export HOME=$PWD

export XAUTHORITY=$HOME/.Xauthority
export DISPLAY=:$((100 + RANDOM % 154))

#export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LD_LIBRARY_PATH=''

xpra --xvfb="Xorg -noreset -nolisten tcp
    -config /etc/xpra/xorg.conf
    -logfile ${HOME}/.xpra/Xorg-${DISPLAY}.log" \
    start $DISPLAY > /dev/null 2>&1
