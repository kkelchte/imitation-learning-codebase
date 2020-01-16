VM=sylabs/singularity-3.2-ubuntu-bionic64
version=$(shell cat VERSION)
LOGINCOMMAND="/bin/bash /vagrant/login_remote.sh"
BUILDCOMMAND="/bin/bash /vagrant/create_image.sh ${version}"

Vagrantfile:
	vagrant init ${VM} && \
	echo 'Adjust memory in Vagrantfile'

spinoff-vagrant: Vagrantfile
	vagrant up
	vagrant ssh -- -t ${LOGINCOMMAND}

shutdown-vagrant:
	vagrant halt

clean-vm:
	vagrant destroy
	rm *.sif
	rm -r .vagrant

*.sif: Vagrantfile VERSION requirements create_image.sh singularity.def
	if [ -e cuda-* ] ; then \
		echo "Found cuda folder: $(echo cuda-*)"; \
		if [ -e cudnn-* ] ; then \
			echo "Found cudnn folder: $(echo cudnn-*)"; \
			vagrant ssh -- -t ${BUILDCOMMAND}; \
		else \
			echo "ERROR: Could not found cudnn folder."; \
		fi ; \
	else \
		echo "ERROR: Could not found cuda folder."; \
	fi

push-image:
	vagrant ssh -- -t ${BUILDCOMMAND}

update-image:
	if [ -e *.sif ] ; then \
		for image in *.sif ; do \
			if [ $$image != ros-gazebo-cuda_${version}.sif ] ; then \
				echo "clean up: $$image"; \
				rm $$image; \
				singularity cache clean --name $$image; \
			fi ; \
		done ; \
	fi
	if [ ! -e ./ros-gazebo-cuda_${version}.sif ] ; then \
		singularity pull library://kkelchte/default/ros-gazebo-cuda:${version} && \
		echo "Successfully pulled singularity-image"; \
	else \
		echo "Singularity image already there"; \
	fi

CODEDIR = ..
test-image: *.sif test/singularity_sanity_check.py
	cd $(CODEDIR) && \
	singularity run --nv singularity/*.sif \
	./launch.sh 'python3.7 singularity/test/singularity_sanity_check.py'

singularity-shell:
	cd $(CODEDIR) && \
	echo "DO NOT FORGET TO SOURCE ENVIRONMENT:" && \
	echo "source entrypoint.sh" && \
	singularity shell --nv singularity/*.sif
