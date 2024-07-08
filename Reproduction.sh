#!/bin/bash
pipenv run python3 main.py -i datasets/kronodroid_real_device-balanced.csv  --num_samples_class_benign 10000 --num_samples\_class_malware 10000 --batch\_size 256 --dense_layer_sizes_g 4096 --dense_layer_sizes_d 2048 --number_epochs 500--k_fold 10
pipenv run python3 main.py -i datasets/kronodroid_real_emulador-balanced.csv  --num_samples_class_benign 10000 --num_samples_class_malware 10000 --batch_size 256 --dense_layer_sizes_g 4096 --dense_layer_sizes_d 2048 --number_epochs 500--k_fold 10
