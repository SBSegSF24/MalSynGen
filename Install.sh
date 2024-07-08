#!/bin/sh
sudo apt-get -y install python3-pip
pip install pipenv
pipenv install -r requirements.txt
