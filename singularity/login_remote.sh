#!/bin/sh

if [ ! -e /home/vagrant/.singularity/remote.yaml ] ; then
	mkdir -p /home/vagrant/.singularity
	touch /home/vagrant/.singularity/remote.yaml
	singularity remote login SylabsCloud
fi
