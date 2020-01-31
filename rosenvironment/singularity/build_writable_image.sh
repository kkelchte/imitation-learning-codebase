#!/bin/bash
echo "
 # $ sudo su
  # # lsblk
  # # fdisk /dev/sda
  # # : p
  # # : n
  # # : p
  # # : 2
  # # : w
  # # lsblk
  # # mount -t ntfs /dev/sda2  /usr/data
"
sudo lsblk
#sudo fdisk /dev/sda
sudo lsblk
sudo mkntfs /dev/sda2
sudo mkdir -p /usr/data
sudo mount -t ntfs /dev/sda2 /usr/data
sudo singularity build --sandbox /usr/data/writable_image /vagrant/*.sif
