#!/bin/bash

#pipenv install -r requirements.txt

pipenv run python3 run_campaign.py -c campanhas_SF24
jupyter notebook plots.ipynb
