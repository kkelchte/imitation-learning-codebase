# Singularity on Mac instructions

### Installation

```bash
$ /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
$ brew cask install virtualbox && \
    brew cask install vagrant && \
    brew cask install vagrant-manager
```

Vagrant accepts only Virtualbox version 6.0.0. Use [these steps](https://zeckli.github.io/en/2016/11/05/use-homebrew-cask-to-downgrad-or-install-en.html) to specify an earlier cask version.  

```bash
# if you run for the first time
export VM=sylabs/singularity-3.2-ubuntu-bionic64 && vagrant init $VM && vagrant up && vagrant ssh
```
Setup singularity login.
Requires login on sylabs container library.
```bash
mkdir $HOME/.singularity
touch $HOME/.singularity/remote.yaml
singularity remote login SylabsCloud
```

### Launch VM to create singularity image

```bash
# copy singularity build file 
vagrant ssh
# go to mounted vagrant directory
cd /vagrant
# build image as 
sudo singularity build image.sif singularity.def
# push image online
singularity sign image.sif
singularity push -U image.sif library://kkelchte/default/ros-gazebo-cuda:v0.0.1
```
