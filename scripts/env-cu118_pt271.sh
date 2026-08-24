#!/bin/bash
set -e

CONDA_P=${1:-"$HOME/miniconda3"}
CONDA_BIN=$CONDA_P/bin
ENV=cu118_pt271
if [ ! -d $CONDA_P/envs/$ENV ]; then
	$CONDA_BIN/conda create --name $ENV python=3.13 -y
fi
ENV_BIN=$CONDA_P/envs/$ENV/bin

$ENV_BIN/pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
$ENV_BIN/pip install opencv-python-headless scikit-learn pyyaml tensorboard easydict addict
$ENV_BIN/pip install medpy nibabel itk simpleitk open3d
$ENV_BIN/pip install nltk
$CONDA_BIN/conda install -n $ENV scipy matplotlib pandas scikit-image jupyter h5py -y
$CONDA_BIN/conda install -n $ENV click tqdm ninja imageio numba -y

echo install ipynb kernel
$CONDA_BIN/conda install -n $ENV -y ipykernel
$ENV_BIN/python -m ipykernel install --user --name $ENV --display-name "py ($ENV)"

# echo (optiional) fix CUDNN_STATUS_NOT_INITIALIZED, reason unknown.
# $ENV_BIN/pip uninstall -y cuda-toolkit cuda-bindings cuda-pathfinder \
#         nvidia-cudnn-cu13 nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
#         nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse \
#         nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx || true
# $ENV_BIN/pip install --force-reinstall torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
