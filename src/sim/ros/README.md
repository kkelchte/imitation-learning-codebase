# Creating a new world:

_make world in gazebo_
Draft a world file in gazebo, save it in gazebo/worlds and modify it.
Adjust GUI camera to:
```xml
<gui fullscreen='0'>
  <camera name='user_camera'>
    <pose frame=''>0 0 50 0 1.57 1.57</pose>
    <view_controller>ortho</view_controller>
    <projection_type>orthographic</projection_type>
  </camera>
</gui>
``` 

_define world config_
Define yml config file in config/world and modify it to specify the terminal conditions.

_test expert_

Run test_ros_environment_integrated

# Structure:

_python2_ros_ws_
catkin ws with ros packages which require python2.7.

_python3_ros_ws_
catkin ws with ros packages compiled with python3.8.
Among one package named imitation-learning-ros-packages containing nodes to interact with ros robots.

_config_
The config directory contains all ros-node-specific configurations which are loaded as ros params at roslaunch.

_gazebo_
This directory contains all gazebo assets: models and world files.

_scripts_
Contain specific shell scripts used to by launched in subprocesses by process_wrappers.py:

- One for launching xpra with no configuration.
- One for launching ros and gazebo with related configuration defined by the arguments of load_ros.launch. Launch configuration specifies:
       - booleans related to whether these rosnodes should be loaded: e.g. gazebo, ...
       - world_name: defines both loaded world_config from config/world as gazebo world file in gazebo/world
       - robot_name: defines robot_config 

_src_
Contains all frontend python code containing ros_environment.py as well as helper functions,
 such as environment generators.

_test_
Contains integrated frontend tests to be run in python singularity environment.

# Installation:

Source ROS environment, assuming you're in a singularity image.
If catkin_packages are not build, it should make it automatically.

```shell script
source entrypoint.sh
```

## install SLAM: DSO
Required singularity image version 0.1.3 or higher.
Install Pangolin:
```shell script
cd $HOME/code/imitation-learning-codebase/src/sim/ros
git clone https://github.com/kkelchte/Pangolin.git
# follow instruction Pangolin to build package
git clone https://github.com/kkelchte/dso.git
# follow instruction dso to build package
cd python3_ws/src
git clone -b catkin https://github.com/kkelchte/dso_ros.git
cd ../..
catkin_make
```
If dso_ros does not want to build, it is probably due to not finding Pangolin or DSO.
Link CMAKE_PATH_PREFIX to the pangolin directory.

# Pycharm:
You can add pycharm sourcing scripts to interactively code and debug in pycharm 
with the correct python interpreter environment.

The ROS frontend code is written in python and expects ROS installation.
Add the specific `pycharm_singularity` alias to your bash script:

```shell script
pycharm_singularity(){
        cd $HOME/code/imitation-learning-codebase
        singularity run --nv singularity/*.sif /bin/bash \
${PATH-TO-PYCHARM-DIR}/applications/pycharm-community-2019.3.1/bin/source_pycharm.sh
}
```

And add the following script to the pycharm bin with name `source_pycharm.sh`:
```shell script
#!/bin/bash
source ./entrypoint.sh
/bin/sh /users/visics/kkelchte/applications/pycharm-community-2019.3.1/bin/pycharm.sh
```

The ROS backend code, containing ros nodes, is written in python2.7.
Add the specific `pycharm_singularity_ros` alias to your bash script:

```shell script
pycharm_singularity_ros(){
        cd $HOME/code/imitation-learning-codebase
        singularity run --nv singularity/*.sif /bin/bash \
${PATH-TO-PYCHARM-DIR}/applications/pycharm-community-2019.3.1/bin/source_pycharm_ros.sh
}
```

And add the following script to the pycharm bin with name `source_pycharm.sh`:
```shell script
#!/bin/bash
source ./entrypoint.sh
source ./src/sim/ros/entrypoint.sh
/bin/sh /users/visics/kkelchte/applications/pycharm-community-2019.3.1/bin/pycharm.sh
```

# Troubleshoot

bebop_driver failed to start as libarcommands not found:
cp -r /opt/ros/melodic/lib/parrot_arsdk/* src/sim/ros/python3_ros_ws/devel/lib