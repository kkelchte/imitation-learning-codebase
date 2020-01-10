#!/bin/sh

if [ ! -e /home/vagrant/.singularity/remote.yaml ] ; then
	mkdir -p /home/vagrant/.singularity
	ln -s /vagrant/remote.yaml /home/vagrant/.singularity/remote.yaml
	#touch /home/vagrant/.singularity/remote.yaml
	#singularity remote login SylabsCloud
fi
