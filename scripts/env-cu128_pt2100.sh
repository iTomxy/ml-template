#!/usr/bin/env bash
set -Eeuo pipefail

# Build this environment separately on cetus, iHPC, and cbai. A single
# environment covers the whole fleet -- sm_70 (iHPC saturn14's V100) through
# sm_120 (cetus large_gpuq's Blackwell) -- but only because of the torch
# version pinned below, so do not bump it casually:
#
#   torch 2.7.x    cu128 = 7.5;8.0;8.6;9.0;10.0;12.0+PTX  -- no sm_70
#   torch 2.8-2.10 cu128 = 7.0;7.5;8.0;8.6;9.0;10.0;12.0  -- sm_70 restored,
#                                              see pytorch/pytorch#157517
#   torch 2.11+    cu128 = 7.5;8.0;8.6;9.0;10.0;12.0      -- sm_70 dropped again
#
# TORCH_CUDA_ARCH_LIST cannot paper over a gap there: it governs only the ops
# this project compiles, while the missing kernels would be torch's own, and
# these cu128 wheels ship no PTX to JIT from. The post-install check below
# verifies the wheel really carries every architecture in the fleet, so a bump
# out of the 2.8-2.10 window fails here instead of on saturn14.
#
# CUDA 12.8 is pinned at both ends: iHPC's driver 570.144 supports no higher,
# and 12.8 is the earliest release that compiles Blackwell (sm_120) while still
# accepting Volta (sm_70), which CUDA 13 removes outright.
#
# PyTorch 2.10 requires GCC >=9 and CUDA 12.8 supports GCC through 14, so let
# conda solve within that range and choose the compiler's sysroot/binutils.
CONDA_PREFIX_ROOT=${1:-"${CONDA_PREFIX_ROOT:-${HOME}/miniconda3}"}
ENV_NAME=${ENV_NAME:-cu128_pt2100}
PYTHON_VERSION=${PYTHON_VERSION:-3.13}
TORCH_VERSION=${TORCH_VERSION:-2.10.0}
TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.25.0}
CUDA_VERSION=12.8
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

# Every GPU measured across the fleet, with the nvcc targets that compile for
# them:
#   7.0   iHPC saturn14 Tesla V100 (Volta)
#   7.5   cetus small_gpuq/med_gpuq Quadro RTX 6000 (Turing)
#   8.6   cbai A4000 and iHPC venus7/venus11 A5500 (Ampere)
#   8.9   iHPC mars3/mars4 L4 and saturn2 L40 (Ada)
#   12.0  cetus large_gpuq RTX PRO 6000 Blackwell
FLEET_DEVICE_ARCHS=(7.0 7.5 8.6 8.9 12.0)
REQUIRED_NVCC_ARCHS=(compute_70 compute_75 compute_86 compute_89 compute_120)

CONDA_BIN="${CONDA_PREFIX_ROOT}/bin/conda"
ENV_ROOT="${CONDA_PREFIX_ROOT}/envs/${ENV_NAME}"
ENV_BIN="${ENV_ROOT}/bin"
PIP_CACHE_DIR=${PIP_CACHE_DIR:-"${CONDA_PREFIX_ROOT}/pkgs/pip-cache"}
export PIP_CACHE_DIR

if [[ ! -x "${CONDA_BIN}" ]]; then
    echo "Conda executable not found at ${CONDA_BIN}" >&2
    exit 1
fi

# An interrupted conda transaction leaves the prefix behind in one of two
# shapes: without bin/python, or -- after a failed create -- with bin/python but
# an empty conda-meta. conda does not recognise the second as an environment, so
# the `conda install` path below aborts with DirectoryNotACondaEnvironmentError
# (hit on iHPC venus25, 2026-08-29). conda-meta/history is the marker conda
# itself looks for; testing the file rather than running `conda list` keeps this
# instant on a network home, where a conda subprocess costs minutes.
if [[ -e "${ENV_ROOT}" ]] \
    && { [[ ! -x "${ENV_BIN}/python" ]] \
        || [[ ! -f "${ENV_ROOT}/conda-meta/history" ]]; }; then
    echo "Incomplete or unrecognised conda environment at ${ENV_ROOT}." >&2
    echo "Move or remove that incomplete prefix, then rerun this script." >&2
    exit 1
fi
mkdir -p "${PIP_CACHE_DIR}"

echo "Building ${ENV_NAME} (CUDA ${CUDA_VERSION}, torch ${TORCH_VERSION})"
echo "Target device architectures: ${FLEET_DEVICE_ARCHS[*]}"

# Solve every available base dependency together. NVIDIA must precede
# conda-forge under strict channel priority so the CUDA packages are not
# excluded. Only compatibility boundaries are constrained; conda chooses the
# concrete package versions and the compiler's transitive sysroot/binutils.
#
# Kept deliberately small: this project imports nibabel, medpy, matplotlib and
# scipy directly, and openpoints declares the rest of its runtime needs (tqdm,
# pandas, h5py, scikit-learn, easydict, pyyaml) in its own metadata, so pip
# installs those when scripts/env.sh installs openpoints. The compiler and CUDA
# development packages are not optional -- scripts/env.sh builds the openpoints
# CUDA ops from source, and no GPU node in the fleet has a usable host nvcc.
CONDA_PACKAGES=(
    "python=${PYTHON_VERSION}"
    pip setuptools wheel
    scipy matplotlib jupyter ninja ipykernel
    tensorboard pillow
    medpy nibabel
    "gxx_linux-64>=9,<15"
    # cuda-nvcc must be pinned, not merely constrained by cuda-version: left
    # free, the solver picked nvcc 12.4 on both iHPC nodes (2026-08-29), which
    # cannot target compute_120 and does not match the 12.8 torch wheel.
    "cuda-version=${CUDA_VERSION}" "cuda-nvcc=${CUDA_VERSION}"
    cuda-cudart-dev cuda-driver-dev
    # torch 2.10's ATen/cuda/CUDAContextLight.h includes cusparse.h, cublas_v2.h,
    # cublasLt.h and cusolverDn.h for every CUDA extension build, and
    # torch.utils.cpp_extension searches only this prefix's include tree -- never
    # the pip nvidia-*-cu12 wheels that also ship them. Without these three,
    # every openpoints CUDA-op build in scripts/env.sh dies at
    # "cusparse.h: No such file or directory". The ops themselves include no
    # math-library headers, so this trio is the whole requirement; cuda-toolkit
    # would also work but drags in the entire toolkit.
    libcublas-dev libcusparse-dev libcusolver-dev
)
CONDA_SOLVE_ARGS=(
    --yes
    --prefix "${ENV_ROOT}"
    --override-channels
    --strict-channel-priority
    --channel nvidia
    --channel conda-forge
)

if [[ ! -x "${ENV_BIN}/python" ]]; then
    echo "Creating ${ENV_NAME} at ${ENV_ROOT}..."
    "${CONDA_BIN}" create "${CONDA_SOLVE_ARGS[@]}" "${CONDA_PACKAGES[@]}"
else
    echo "Updating ${ENV_NAME} at ${ENV_ROOT}..."
    "${CONDA_BIN}" install "${CONDA_SOLVE_ARGS[@]}" "${CONDA_PACKAGES[@]}"
fi

# Put this env's own runtime libraries ahead of the system paths. Conda's
# `activate` would do this, but this script drives python/nvcc by absolute path,
# so without it the login node's /lib64/libstdc++.so.6 (which lacks
# GLIBCXX_3.4.29) shadows the env's libstdc++ and numpy/torch fail to import.
export LD_LIBRARY_PATH="${ENV_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# PyTorch's CUDA wheel index is the source of truth for this exact pair.
"${ENV_BIN}/python" -m pip install \
    "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

for executable in ninja nvcc \
    x86_64-conda-linux-gnu-gcc \
    x86_64-conda-linux-gnu-g++ \
    x86_64-conda-linux-gnu-ld; do
    if [[ ! -x "${ENV_BIN}/${executable}" ]]; then
        echo "Base environment is incomplete: ${ENV_BIN}/${executable} is missing." >&2
        exit 1
    fi
done

# nvcc must be able to target every architecture in the fleet, and the check
# must run before any extension build rather than after it.
NVCC_ARCHS="$("${ENV_BIN}/nvcc" --list-gpu-arch)"
for arch in "${REQUIRED_NVCC_ARCHS[@]}"; do
    if ! grep -qx -- "${arch}" <<<"${NVCC_ARCHS}"; then
        echo "nvcc cannot target ${arch}; the CUDA ${CUDA_VERSION} compiler is incomplete." >&2
        exit 1
    fi
done

if [[ ! -f "${ENV_ROOT}/targets/x86_64-linux/include/cuda.h" \
    || ! -f "${ENV_ROOT}/targets/x86_64-linux/include/cuda_runtime.h" ]]; then
    echo "Base environment is incomplete: CUDA ${CUDA_VERSION} headers are missing." >&2
    exit 1
fi
if [[ ! -e "${ENV_ROOT}/targets/x86_64-linux/lib/libcudart.so" ]]; then
    echo "Base environment is incomplete: CUDA ${CUDA_VERSION} development library is missing." >&2
    exit 1
fi

# The math-library headers torch's CUDAContextLight.h pulls into every extension
# build. Checking them here turns a channel change into a clear failure of this
# script, instead of "cusparse.h: No such file or directory" much later, inside
# the first openpoints CUDA-op compile in scripts/env.sh.
for header in cusparse.h cublas_v2.h cublasLt.h cusolverDn.h; do
    if [[ ! -f "${ENV_ROOT}/targets/x86_64-linux/include/${header}" ]]; then
        echo "Base environment is incomplete: ${header} is missing." >&2
        echo "libcublas-dev, libcusparse-dev and libcusolver-dev supply these;" >&2
        echo "if the channels stop carrying them, use cuda-toolkit=${CUDA_VERSION}." >&2
        exit 1
    fi
done

TORCH_CUDA_VERSION="$("${ENV_BIN}/python" -c 'import torch; print(torch.version.cuda)')"
if [[ "${TORCH_CUDA_VERSION}" != "${CUDA_VERSION}" ]]; then
    echo "Expected PyTorch CUDA ${CUDA_VERSION}, found ${TORCH_CUDA_VERSION}." >&2
    exit 1
fi

# The wheel's own kernels decide which nodes this environment can run on, and
# no later setting can change them. Reject a wheel that cannot serve one of the
# fleet architectures here, rather than at "no kernel image is available for
# execution on the device" hours into a training run. This is the check that
# pins torch to the 2.8-2.10 window: 2.7.x and 2.11+ drop sm_70.
"${ENV_BIN}/python" - "${FLEET_DEVICE_ARCHS[@]}" <<'PY'
import sys

import torch

# Private, but the only way to read the compiled kernels without a GPU:
# torch.cuda.get_arch_list() returns [] when no device is visible.
getter = getattr(torch._C, "_cuda_getArchFlags", None)
if getter is not None:
    flags = getter() or ""
elif torch.cuda.is_available():
    flags = " ".join(torch.cuda.get_arch_list())
else:
    raise SystemExit(
        "Cannot read this torch build's architectures: torch "
        f"{torch.__version__} has no _cuda_getArchFlags and no GPU is visible."
    )

cubins, ptx = [], []
for flag in flags.split():
    kind, _, digits = flag.partition("_")
    if not digits.isdigit():
        continue
    capability = (int(digits[:-1]), int(digits[-1]))
    (cubins if kind == "sm" else ptx).append(capability)

print(f"torch {torch.__version__} kernels: {flags}")

missing = []
for argument in sys.argv[1:]:
    major, minor = (int(part) for part in argument.split("."))
    # A cubin runs on a device of the same major version and an equal or higher
    # minor version; PTX JITs forward to any newer architecture, never back.
    if any(c[0] == major and c[1] <= minor for c in cubins):
        continue
    if any(p <= (major, minor) for p in ptx):
        continue
    missing.append(argument)

if missing:
    raise SystemExit(
        f"torch {torch.__version__} carries no kernels for compute capability "
        f"{', '.join(missing)}; this wheel cannot serve every target device."
    )
PY

# Install (or refresh) the user-visible Jupyter kernel for this prefix.
"${ENV_BIN}/python" -m ipykernel install --user \
    --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

"${ENV_BIN}/python" -m pip check
"${ENV_BIN}/python" -c \
    'import torch, torchvision; print("ready:", torch.__version__, torchvision.__version__, "CUDA", torch.version.cuda)'
"${ENV_BIN}/nvcc" --version | tail -n 1
