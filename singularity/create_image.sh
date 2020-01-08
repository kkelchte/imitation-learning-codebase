#!/bin/sh
VERSION=$1
if [ -z "${VERSION}" ] ; then
	VERSION="latest"
fi
echo "VERSION = ${VERSION}"

cd /vagrant || exit

if [ -e image-${VERSION}.sif ] ; then
	echo "found existing singularity image: $(echo ./*.sif)"
else
	sudo singularity build image-${VERSION}.sif singularity.def
fi

if [ -e image-${VERSION}.sif ] ; then
  singularity sign image-${VERSION}.sif
  echo "singularity push image-${VERSION}.sif library://kkelchte/default/ros-gazebo-cuda:${VERSION} ..."
  # TODO: -U is required as container verification takes a long time and results in a connection error
  singularity push -U image-${VERSION}.sif library://kkelchte/default/ros-gazebo-cuda:${VERSION}
fi
