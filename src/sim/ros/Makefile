
catkin_ws/src:
	mkdir -p catkin_ws/src

catkin_ws/devel/setup.bash: catkin_ws/src
	cd catkin_ws && \
	catkin_make

install: catkin_ws/src
	cd catkin_ws && \
	catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.7m \
	-DPYTHON_LIBRARY=/usr/lib/libpython3.7m.so
