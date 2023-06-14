#!/bin/bash

## Calling
# . find-gpu.sh [#gpu required] [min memory per gpu] [b/w]
## Input
# $1: # of GPU needed
# $2: lower bound of free memory needed per GPU (in MiB)
## Output
# gpu_id: int/str, selected GPU id (list, seperated by ','),
#	default = -1, can be fed to `CUDA_VISIBLE_DEVICES`.
# n_gpu_found: int, # of selected GPU.
## Reference
# https://blog.csdn.net/HackerTom/article/details/126257508

# 1st arg: # of GPU needed, default = 1
#	<0 means requiring all GPUs.
_n_gpu_req=${1-"1"}
# 2nd arg: lower bound of free memory needed per GPU (in MiB), default = 0
#	i.e. only GPUs with free memory >= $2 will be selected.
_mem_lb=${2-"0"}
# 3rd arg: fit mode, in {b, w}, default = b
#	b: best fit, prioritises those have LEAST but enough available memory
#	w: worst fit, prioritises those have MOST and enough available memory
_mode=${3-"b"}

_res=$(nvidia-smi | \
	grep -E "[0-9]+MiB\s*/\s*[0-9]+MiB" | \
	sed "s/^|//" | \
	awk '{print ($8" "$10)}' | \
	sed "s/\([0-9]\+\)MiB \([0-9]\+\)MiB/\1 \2/" | \
	awk '{print $2 - $1}')

_i=0
if [ $_mode == "b" ]; then
	_res=($(for _s in $_res; do echo $_i $_s && _i=`expr 1 + $_i`; done | \
		sort -n -k 2))
else
	_res=($(for _s in $_res; do echo $_i $_s && _i=`expr 1 + $_i`; done | \
		sort -n -k 2 -r))
fi

if [ ${_n_gpu_req} -lt 0 ]; then
	# echo using all GPUs
	_n_gpu_req=`expr ${#_res[@]} / 2`
	# echo ${n_gpu_req}
fi

gpu_id=-1
n_gpu_found=0
for i in $(seq 0 2 `expr ${#_res[@]} - 1`); do
	_gid=${_res[i]}
	_mem=${_res[i+1]}
	# echo $_gid: $_mem
	if [ ${n_gpu_found} -lt ${_n_gpu_req} -a ${_mem} -ge ${_mem_lb} ]; then
		if [ ${n_gpu_found} -eq 0 ]; then
			gpu_id=${_gid}
		else
			gpu_id=${gpu_id}","${_gid}
		fi
		n_gpu_found=`expr 1 + ${n_gpu_found}`
	# else
	#	break
	fi
done

# echo found: ${n_gpu_found}

