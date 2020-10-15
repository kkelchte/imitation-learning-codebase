version=$(shell cat VERSION)
CODEDIR=../..

create-sif-linux:
	sudo singularity build image-${version}.sif singularity.def

shell-writable-image-linux:
	singularity shell --bind ../..:/mnt --bind /tmp/.X11-unix:/tmp/.X11-unix --writable writable_image

create-writable-image-linux:
	sudo singularity build --sandbox writable_image image-${version}.sif

build-writable-to-sif-linux:
	sudo singularity build image-${version}.sif writable_image


push-image: spinoff-vagrant
	vagrant ssh -- -t ${BUILDCOMMAND}

# Following commands are for on linux machines without vagrant.

update-image:
	if [ -e *.sif ] ; then \
		for image in *.sif ; do \
			if [ $$image != ros-gazebo-cuda_${version}.sif ] ; then \
				echo "clean up: $$image"; \
				mv $$image /esat/opal/kkelchte/singularity_images; \
#				rm $$image; \
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

test-image: *.sif test/singularity_sanity_check.py
	cd $(CODEDIR) && \
	singularity run --nv rosenvironment/singularity/*.sif \
	./rosenvironment/entrypoint.sh 'python3.8 rosenvironment/singularity/test/singularity_sanity_check.py'

singularity-shell:
	cd $(CODEDIR) && \
	echo "DO NOT FORGET TO SOURCE ENVIRONMENT:" && \
	echo "source rosenvironment/entrypoint.sh" && \
	singularity shell --nv rosenvironment/singularity/image-${version}.sif
