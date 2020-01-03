VM=sylabs/singularity-3.2-ubuntu-bionic64
VERSION=$(shell cat VERSION)

COMMAND="/bin/bash /vagrant/create_image.sh ${VERSION}"

launch-singularity:
	if [ ! -e *.sif ] ; then \
		singularity pull

create-vm:
	vagrant init ${VM}
	vagrant up

clean-vm:
	vagrant destroy
	rm Vagrantfile
	rm -r .vagrant

build-image: #create-vm
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