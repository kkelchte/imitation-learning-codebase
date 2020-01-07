#!/bin/sh
VERSION=$1
if [ -z "${VERSION}" ] ; then
	VERSION="latest"
fi
echo "VERSION = ${VERSION}"

cd /vagrant || exit

if [ -e image.sif ] ; then
	echo "found existing singularity image: $(echo ./*.sif)"
else
	sudo singularity build image.sif singularity.def
fi

if [ ! -e /home/vagrant/.singularity/remote.yaml ] ; then
	mkdir -p /home/vagrant/.singularity
	touch /home/vagrant/.singularity/remote.yaml
	singularity remote login SylabsCloud
fi

if [ -e image.sif ] ; then
  singularity sign image.sif
  echo "singularity push image.sif library://kkelchte/default/ros-gazebo-cuda:${VERSION} ..."
  # TODO: -U is required as container verification takes a long time and results in a connection error
  singularity push -U image.sif library://kkelchte/default/ros-gazebo-cuda:${VERSION}
fi
