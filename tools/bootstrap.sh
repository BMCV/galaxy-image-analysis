#!/bin/bash

echo "Provisioning virtual machine..."

#echo "Upgrade VM..."
#apt-get update
#apt-get upgrade -y openssl wget

echo "Install base software..."
apt-get install -y nano curl git software-properties-common
sudo apt-add-repository universe
apt-get update
apt-get install -y python-setuptools python-dev build-essential python-pip

pip install --upgrade pip wheel virtualenv

echo "Install Planemo..."
su - vagrant
pip install planemo

echo "...finished provisioning!"
