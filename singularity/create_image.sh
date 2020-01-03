#!/bin/sh
cd /vagrant

sudo singularity build image.sif singularity.def

singularity push -U image.sif library://kkelchte/default/ros-gazebo-cuda:v0.0.1