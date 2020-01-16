# Instructions

### Get shell in singularity image

```bash
make launch-image
```

### Writable image (sandbox)

__Create__

```bash
vagrant up
vagrant ssh
$ sudo singularity build --sandbox writable_image/ /vagrant/*.sif
```

__Add package__

```bash
vagrant up
vagrant ssh
$ sudo singularity shell --writable writable_image
> sudo su
# apt update
# apt-cache search ros-melodic | grep ${PACKAGE-TAG}
# apt install -qy ${PACKAGE}
CTR+D
> CTR+D
$ sudo singularity build /vagrant/image-${VERSION}.sif writable_image
CTR+D
```


### Create singularity image on Mac from scratch

__Makefile__

Create Vagrant environment, build singularity image and push it to singularity library.
Dependency: cudnn-* and cuda-* libraries should be in current directory.

```bash
make *.sif
```


__Manually__

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

__Cleanup directory__

Should only be done if vagrant VM should be destroyed.

```bash
make clean-vm
```


### Installation of VB and Vagrant on Mac

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

