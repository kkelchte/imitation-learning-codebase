ENVIRONMENT_DIR=venv

.PHONY: venv

venv: requirements
	if [ -d ${ENVIRONMENT_DIR} ] ; then \
		rm -r ${ENVIRONMENT_DIR}; \
	fi
	virtualenv -p python3 ${ENVIRONMENT_DIR}
	@source venv/bin/activate \
	&& pip install -r requirements
	echo 'Dont forget to update python interpreter in pycharm'

