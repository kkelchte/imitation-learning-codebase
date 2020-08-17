install: install_python3_ros_ws install_python2_ros_ws

#################
# python3_ros_ws
#################
python3_ros_ws/src:
	mkdir -p python3_ros_ws/src && \
	cd python3_ros_ws && \
	catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3.8 -DPYTHON_INCLUDE_DIR=/usr/include/python3.8 \
	-DPYTHON_LIBRARY=/usr/lib/python3.8


install_python3_ros_ws: python3_ros_ws/src
	cd python3_ros_ws && \
	git clone https://github.com/kkelchte/bebop_autonomy.git src/bebop_autonomy && \
	git clone https://github.com/kkelchte/vision_opencv.git src/vision_opencv && \
	catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3.8 -DPYTHON_INCLUDE_DIR=/usr/include/python3.8 \
	-DPYTHON_LIBRARY=/usr/lib/python3.8 && \
	catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3.8 -DPYTHON_INCLUDE_DIR=/usr/include/python3.8 \
	-DPYTHON_LIBRARY=/usr/lib/python3.8
	# Build twice because bebop autonomy has some dependencies on itself which requires a rebuild


python3_ros_ws/devel/setup.bash: $(wildcard python3_ros_ws/src/**/*)
	cd python3_ros_ws && \
	for f in src/imitation_learning_ros_package/rosnodes/*.py ; do chmod u+x $$f; done && \
	catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3.8 \
	-DPYTHON_INCLUDE_DIR=/usr/include/python3.8 -DPYTHON_LIBRARY=/usr/lib/python3.8


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


#################
# Test ROS environment
#################

test_suite: test/test_*.py
	cd ${HOME} && date &&\
	if [ -d suite_output_test ] ; then rm -r suite_output_test; fi && mkdir suite_output_test &&\
	for f in $^ ; do echo $${f}; python3.8 src/sim/ros/$${f} > suite_output_$${f} 2>&1; sleep 5; \
	tail suite_output_$${f}; done && cd src/sim/ros && date

