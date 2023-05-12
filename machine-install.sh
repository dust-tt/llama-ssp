#!/bin/bash
apt update
apt install -y python3-pip python3-venv

# cuda install
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring*.deb
apt-get update
apt-get -y install cuda



