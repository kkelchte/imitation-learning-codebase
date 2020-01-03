#!/bin/sh
cd /vagrant
echo "VERSION: $1"

sudo singularity build image.sif singularity.def

singularity push -U image.sif library://kkelchte/default/ros-gazebo-cuda:$1