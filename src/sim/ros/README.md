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

_catkin_ws_
All ROS nodes and launch files are stored in the catkin_ws/src/imitation_learning_ros_package,
 as well as other ROS dependencies which are not installed system wide.
The nodes are written in python2.7 and serve as backend.
Corresponding tests are found in catkin_ws/src/imitation_learning_ros_package/test and should be run with python2.7 
in a singularity - ros environment.

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