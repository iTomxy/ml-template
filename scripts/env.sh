#!/bin/bash
set -e

CONDA_P=${1-"$HOME/miniconda3"}
CONDA_BIN=$CONDA_P/bin
ENV=stylehuman
if [ ! -d $CONDA_P/envs/$ENV ]; then
    $CONDA_BIN/conda create --name $ENV python=3.8 -y
fi
ENV_BIN=$CONDA_P/envs/$ENV/bin

echo PyTorch
conda install -n $ENV pytorch=1.10.1=*cuda* cudatoolkit=11.3 cudnn=8.2 -c pytorch -c nvidia -y
$ENV_BIN/pip install torchvision==0.11.2+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

echo misc
conda install -n $ENV click scipy tqdm ninja matplotlib imageio pandas -y
for pkg in \
    imageio-ffmpeg opencv-python-headless omegaconf einops transformers \
    imgui glfw pyopengl lpips pyspng dlib moviepy imutils wandb \
    scikit-learn simpleitk itk nltk tensorboard
do
    $ENV_BIN/pip install $pkg
done

echo paddle
$ENV_BIN/python -m pip install paddlepaddle-gpu==2.4.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
cd $HOME/codes
if [ ! -d PaddleSeg ]; then
    git clone https://github.com/PaddlePaddle/PaddleSeg --branch release/2.5
fi
cd PaddleSeg
$ENV_BIN/pip install -r requirements.txt
$ENV_BIN/pip install -e .
cd Matting
$ENV_BIN/pip install -r requirements.txt

