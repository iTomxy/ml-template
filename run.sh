#!/bin/bash

# clear

echo "hello world!"
python main.py --data_path ~/data/data20371/ --cnnf_weight ~/data/data20371/vgg_net.mat

# for i_fold in 0 1 2 3 4; do
#     python main.py --data_path ~/data/data20371/ --cnnf_weight ~/data/data20371/vgg_net.mat --i_fold $i_fold --n_fold $n_fold
#     if [ $? -ne 0 ]; then exit; fi
# done
