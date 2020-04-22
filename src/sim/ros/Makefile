install: install_catkin_ws install_python3_ros_ws install_python2_ros_ws

############
# Catkin_ws
############
catkin_ws/src:
	mkdir -p catkin_ws/src

install_catkin_ws: catkin_ws/src
	cd catkin_ws && \
	catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3.7 -DPYTHON_INCLUDE_DIR=/usr/include/python3.7m \
	-DPYTHON_LIBRARY=/usr/lib/libpython3.7m.so

catkin_ws/devel/setup.bash: $(wildcard catkin_ws/src/**/*)
	cd catkin_ws && \
	for f in src/imitation_learning_ros_package/rosnodes/*.py ; do chmod u+x $$f; done && \
	catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3.7 \
	-DPYTHON_INCLUDE_DIR=/usr/include/python3.7m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.7m.so

#################
# python3_ros_ws
#################
python3_ros_ws/src:
	mkdir -p python3_ros_ws/src && \
	cd python3_ros_ws && \
	catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3.7 -DPYTHON_INCLUDE_DIR=/usr/include/python3.7m \
	-DPYTHON_LIBRARY=/usr/lib/libpython3.7m.so


install_python3_ros_ws: python3_ros_ws/src
	cd python3_ros_ws && \
	git clone https://github.com/AutonomyLab/bebop_autonomy.git src/bebop_autonomy && \
	git clone https://github.com/ros-perception/vision_opencv.git src/vision_opencv && \
	catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3.7 -DPYTHON_INCLUDE_DIR=/usr/include/python3.7m \
	-DPYTHON_LIBRARY=/usr/lib/libpython3.7m.so && \
	catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3.7 -DPYTHON_INCLUDE_DIR=/usr/include/python3.7m \
	-DPYTHON_LIBRARY=/usr/lib/libpython3.7m.so
	# Build twice because bebop autonomy has some dependencies on itself which requires a rebuild


python3_ros_ws/devel/setup.bash: $(wildcard python3_ros_ws/src/**/*)
	cd python3_ros_ws && \
	catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3.7 \
	-DPYTHON_INCLUDE_DIR=/usr/include/python3.7m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.7m.so

#################
# Python2_ros_ws
#################

python2_ros_ws/src:
	mkdir -p python2_ros_ws/src && \
	cd python2_ros_ws && \
	catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python2.7 -DPYTHON_INCLUDE_DIR=/usr/include/python2.7m \
	-DPYTHON_LIBRARY=/usr/lib/libpython2.7m.so

install_python2_ros_ws: python2_ros_ws/src
	cd python2_ros_ws && \
	git clone https://github.com/kkelchte/hector_quadrotor.git src/hector_quadrotor && \
	git clone https://github.com/tu-darmstadt-ros-pkg/hector_localization.git src/hector_localization && \
	catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python2.7 -DPYTHON_INCLUDE_DIR=/usr/include/python2.7m \
	-DPYTHON_LIBRARY=/usr/lib/libpython2.7m.so

python2_ros_ws/devel/setup.bash: $(wildcard python2_ros_ws/src/**/*)
	cd python2_ros_ws && \
	catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python2.7 \
	-DPYTHON_INCLUDE_DIR=/usr/include/python2.7m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython2.7m.so
