# Installation:

Source ROS environment, assuming you're in a singularity image.
If catkin_packages are not build, it should make it automatically.

```shell script
source entrypoint.sh
```

# Pycharm:
You can add pycharm sourcing scripts to interactively code and debug in pycharm 
with the correct python interpreter environment.

The ROS frontend code is written in python3.7 and expects ROS installation.
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