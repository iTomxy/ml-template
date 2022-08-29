#!/bin/bash

## Calling
# . <PATH PREFIX>/find_gpu.sh
# . <PATH PREFIX>/find_gpu.sh 5
# . <PATH PREFIX>/find_gpu.sh -1
# . <PATH PREFIX>/find_gpu.sh 3 8000
# . <PATH PREFIX>/find_gpu.sh -1 20480
## Input
# $1: # of GPU needed
# $2: lower bound of free memory needed per GPU (in MiB)
## Output
# gpu_id: int/str, selected GPU id (list, seperated by ','),
#   default = -1, can be fed to `CUDA_VISIBLE_DEVICES`.
# n_gpu_found: int, # of selected GPU.
## Reference
# https://blog.csdn.net/HackerTom/article/details/126257508

# 1st arg: # of GPU needed, default = 1
#   <0 means requiring all GPUs.
n_gpu_req=${1-"1"}
# 2nd arg: lower bound of free memory needed per GPU (in MiB), default = 0
#   i.e. only GPUs with free memory >= $2 will be selected.
mem_lb=${2-"0"}
# echo "Requiring ${n_gpu_req} GPUs with at least ${mem_lb}MB free memory"

res=$(nvidia-smi | \
	grep -E "[0-9]+MiB\s*/\s*[0-9]+MiB" | \
	awk '{print ($9" "$11)}' | \
	sed "s/\([0-9]\{1,\}\)MiB \([0-9]\{1,\}\)MiB/\1 \2/" | \
	awk '{print $2 - $1}')

i=0
res=($(for s in $res; do echo $i $s && i=`expr 1 + $i`; done | \
	sort -n -k 2 -r))

if [ ${n_gpu_req} -lt 0 ]; then
	# echo using all GPUs
	n_gpu_req=`expr ${#res[@]} / 2`
	# echo ${n_gpu_req}
fi

gpu_id=-1
n_gpu_found=0
for i in $(seq 0 2 `expr ${#res[@]} - 1`); do
	gid=${res[i]}
	mem=${res[i+1]}
	# echo $gid: $mem
	if [ ${n_gpu_found} -lt ${n_gpu_req} -a $mem -ge ${mem_lb} ]; then
		if [ ${n_gpu_found} -eq 0 ]; then
			gpu_id=$gid
		else
			gpu_id=${gpu_id}","$gid
		fi
		n_gpu_found=`expr 1 + ${n_gpu_found}`
	else
		break
	fi
done

# echo found: ${n_gpu_found}
