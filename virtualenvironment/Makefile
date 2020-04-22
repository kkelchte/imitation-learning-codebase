ENVIRONMENT_DIR=venv

.PHONY: venv

venv: ../rosenvironment/singularity/requirements
	virtualenv -p python3 ${ENVIRONMENT_DIR}
#	&& pip install -r ../rosenvironment/singularity/requirements
	echo 'source venv/bin/activate and pip install -r ../rosenvironment/singularity/requirements'
