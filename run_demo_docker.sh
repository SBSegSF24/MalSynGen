#!/bin/bash
#USER_ID=$(id -u $USER)
DIR=$(readlink -f shared)
docker build -t sf24/malsyngen:latest . 
docker run -it --name=MalSynGen-$RANDOM -v $DIR:/SynTabData/shared -e DISPLAY=unix$DISPLAY sf24/malsyngen:latest /MalSynGen/shared/app_run.sh --verbosity 20 --output_dir /MalSynGen/shared/outputs --input_dataset /MalSynGen/datasets/kronodroid_emulador-balanced.csv --num_samples_class_malware  10000 --num_samples_class_benign 10000 --batch_size 256   --number_epochs 300 --k_fold 10 --training_algorithm Adam
#ls shared/outputs/
