# imitation-learning-codebase installation guide

## get the codebase

```bash
mkdir -p $HOME/code
cd $HOME/code 
git clone git@github.com:kkelchte/imitation-learning-codebase.git
cd imitation-learning-codebase
```

## Install conda python environment
In the conda virtual environment you can run all code except the ROS-GAZEBO simulator.
If working on Esat, install conda environment on Opal: /esat/opal/r-number/conda.
This directory is specified during the installation process

```bash
# from codebase root directory
cd virtualenvironment
make 
conda init bash
# in new terminal: avoid conda base env to start at each bash
conda config --set auto_activate_base false
# create venv
make create_env
conda activate venv
make install_packages
```

Place some environment variables in .bashrc:

```
echo 'alias cdcodebase="cd ~/code/imitation-learning-codebase && conda activate venv && export PYTHONPATH=~/code/imitation-learning-codebase"' >> $HOME/.bashrc
mkdir -p ~/code/imitation-learning-codebase/experimental_data
echo 'export DATADIR=~/code/imitation-learning-codebase/experimental_data'  >> $HOME/.bashrc
echo 'export CODEDIR=~/code/imitation-learning-codebase'  >> $HOME/.bashrc
# from a new terminal window
cdcodebase
/bin/bash test_suite.sh
```
The condor test will fail if the singularity image (*.sif) is not downloaded yet. Don't worry, just go to the next step.

## Install ROS Gazebo environment

Assuming virtual environment installation is already done.

__copy singularity image locally__

```bash
cd rosenvironment/singularity
scp -r r-number@ssh.esat.kuleuven.be:/users/visics/kkelchte/code/imitation-learning-codebase/rosenvironment/singularity/image-$(cat VERSION).sif .
make singularity-shell
source /opt/ros/melodic/setup.bash
```

__install catkin ROS packages__

```bash
# if you are in a singularity image, skip first three lines
cd rosenvironment/singularity
make singularity-shell
source rosenvironment/entrypoint.sh
# make sure all builds are successfull otherwise perform them again manually. See troubleshoot in src/sim/ros/README.md
# test installation
cd src/sim/ros
make test_suite
```

