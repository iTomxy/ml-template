#!/bin/bash

# clear

echo "hello world!"
# python main.py --data_path ~/data/data20371/ --cnnf_weight ~/data/data20371/vgg_net.mat

n_fold=5
for i_fold in $(seq 0 `expr $n_fold - 1`); do
    python main.py --data_path ~/data/data20371/ --cnnf_weight ~/data/data20371/vgg_net.mat --i_fold $i_fold --n_fold $n_fold
    if [ $? -ne 0 ]; then exit; fi
done
