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

singularity-shell:
	cd $(CODEDIR) && \
	echo "DO NOT FORGET TO SOURCE ENVIRONMENT:" && \
	echo "source virtualenvironment/entrypoint.sh" && \
	singularity shell --bind /tmp/.X11-unix:/tmp/.X11-unix --nv virtualenvironment/singularity/image-${version}.sif
