#!/bin/sh
RUN apt-get -y install python3-pip
RUN pip install pipenv
RUN pipenv install -r requirements.txt
