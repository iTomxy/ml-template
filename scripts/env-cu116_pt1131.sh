#!/bin/bash
set -e

CONDA_P=${1-"$HOME/miniconda3"}
CONDA_BIN=$CONDA_P/bin
ENV=cu116_pt1131
if [ ! -d $CONDA_P/envs/$ENV ]; then
        $CONDA_BIN/conda create --name $ENV python=3.8 -y
fi
ENV_BIN=$CONDA_P/envs/$ENV/bin

$ENV_BIN/pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
$ENV_BIN/pip install opencv-python-headless scikit-learn pyyaml tensorboard easydict
$ENV_BIN/pip install medpy nibabel itk simpleitk monai
$ENV_BIN/pip install nltk
$CONDA_BIN/conda install -n $ENV scipy matplotlib pandas scikit-image jupyter h5py -y
$CONDA_BIN/conda install -n $ENV click tqdm ninja imageio numba -y
