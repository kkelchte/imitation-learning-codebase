ENVIRONMENT_DIR=venv

.PHONY: venv

venv: ../rosenvironment/singularity/requirements
	virtualenv -p python3.7 ${ENVIRONMENT_DIR}
#	&& python3.7 -m pip install -r ../rosenvironment/singularity/requirements
	echo 'source venv/bin/activate and python3.7 -m pip install -r ../rosenvironment/singularity/requirements'
