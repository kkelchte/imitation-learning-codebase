ENVIRONMENT_DIR=venv

.PHONY: venv

venv: ../rosenvironment/singularity/requirements
	virtualenv -p python3 ${ENVIRONMENT_DIR}
	@source venv/bin/activate \
	&& pip install -r ../rosenvironment/singularity/requirements
	echo 'Dont forget to update python interpreter in pycharm'
