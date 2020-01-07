VM=sylabs/singularity-3.2-ubuntu-bionic64
version=$(shell cat VERSION)

COMMAND="/bin/bash /vagrant/create_image.sh ${version}"

Vagrantfile:
	vagrant init ${VM}
	vagrant up

clean-vm:
	vagrant destroy
	rm image.sif
	rm Vagrantfile
	rm -r .vagrant

image.sif: Vagrantfile VERSION requirements create_image.sh singularity.def
	if [ -e cuda-* ] ; then \
		echo "Found cuda folder: $(echo cuda-*)"; \
		if [ -e cudnn-* ] ; then \
			echo "Found cudnn folder: $(echo cudnn-*)"; \
			vagrant ssh -- -t ${COMMAND}; \
		else \
			echo "ERROR: Could not found cudnn folder."; \
		fi ; \
	else \
		echo "ERROR: Could not found cuda folder."; \
	fi

fetch-image:
	if [ ! -e ./*.sif ] ; then \
		singularity pull library://kkelchte/default/ros-gazebo-cuda:${version} && \
		echo "Successfully pulled singularity-image"; \
	else \
		echo "Singularity image already there"; \
	fi
