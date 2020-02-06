#!/bin/bash
echo "
 # $ sudo su
  # check if partition is there under /dev/sda2
  lsblk
  # if partition is not there, create it
  fdisk /dev/sda
  # # : p
  # # : n
  # # : p
  # # : 2
  # # : w
  # check if partition is there under /dev/sda2
  lsblk
  # initialize partition with ntfs
  mkntfs /dev/sda2
  mkdir -p /usr/data
  mount -t ntfs /dev/sda2  /usr/data
"
#sudo lsblk
#sudo fdisk /dev/sda
#sudo lsblk
#sudo mkntfs /dev/sda2
#sudo mkdir -p /usr/data
#sudo mount -t ntfs /dev/sda2 /usr/data
sudo singularity build --sandbox /usr/data/writable_image /vagrant/*.sif
