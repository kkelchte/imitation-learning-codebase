ENVIRONMENT_DIR=venv

.PHONY: venv

venv: ../rosenvironment/singularity/requirements
	if [ -d venv ] ; then rm -r venv; fi && \
	virtualenv -p python3 ${ENVIRONMENT_DIR} \
	echo 'source venv/bin/activate and python3 -m pip install -r ../rosenvironment/singularity/requirements'
