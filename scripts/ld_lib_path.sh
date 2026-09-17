#!/usr/bin/env bash

# Prepend the active environment's lib directory to LD_LIBRARY_PATH.
# Usage:
#   . "$HOME/ld_lib_path.sh"                       # use the activated env
#   LLP_ENV=cu128_pt2100 . "$HOME/ld_lib_path.sh"  # or name one
#   LLP_ENV=... LLP_CONDA_P=/opt/conda . "$HOME/ld_lib_path.sh"
#
# Input arrives as an assignment prefix rather than as $1: a sourced script
# invoked with no arguments inherits the caller's positional parameters, so a
# runner taking its own arguments would have $1 read as an environment name.

llp_env=${LLP_ENV:-}
llp_conda_p=${LLP_CONDA_P:-"$HOME/miniconda3"}

if [[ -n $llp_env ]]; then
    llp_root="${llp_conda_p}/envs/${llp_env}"
elif [[ -n ${CONDA_PREFIX:-} ]]; then
    llp_root=$CONDA_PREFIX
elif [[ -n ${VIRTUAL_ENV:-} ]]; then
    llp_root=$VIRTUAL_ENV
elif [[ -n ${CONDA_DEFAULT_ENV:-} ]]; then
    llp_root="${llp_conda_p}/envs/${CONDA_DEFAULT_ENV}"
else
    echo "No environment specified or active." >&2
    return 1
fi

llp_lib="${llp_root}/lib"

if [[ ":${LD_LIBRARY_PATH:-}:" != *":${llp_lib}:"* ]]; then
    export LD_LIBRARY_PATH="${llp_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
