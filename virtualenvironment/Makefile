ENVIRONMENT_DIR=venv

.PHONY: venv

venv: ../rosenvironment/singularity/requirements
	if [ -d venv ] ; then rm -r venv; fi && \
	virtualenv -p python3.8 ${ENVIRONMENT_DIR} \
#	&& python3.7 -m pip install -r ../rosenvironment/singularity/requirements
	echo 'source venv/bin/activate and python3.8 -m pip install -r ../rosenvironment/singularity/requirements'
