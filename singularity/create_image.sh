#!/bin/sh
cd /vagrant

VERSION=$1
if [ -z $VERSION ] ; then
	VERSION="latest"
fi
echo "VERSION = ${VERSION}"

if [ -e image.sif ] ; then
	echo "found existing singularity image: $(echo *.sif)"
else
	sudo singularity build image.sif singularity.def
fi

if [ ! -e /home/vagrant/.singularity/remote.yaml ] ; then
	mkdir -p /home/vagrant/.singularity
	touch /home/vagrant/.singularity/remote.yaml
	singularity remote login SylabsCloud
fi

if [ -e image.sif ] ; then
  echo "singularity push -U image.sif library://kkelchte/default/ros-gazebo-cuda:${VERSION} ..."
  singularity push -U image.sif library://kkelchte/default/ros-gazebo-cuda:${VERSION}
fi