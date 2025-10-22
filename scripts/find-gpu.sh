#!/bin/bash
# https://blog.csdn.net/HackerTom/article/details/126257508

# reset OPTIND at the start so that `show_help` prints everytime even with `.` execution.
OPTIND=1

show_help() {
	cat << EOF
Usage: . find-gpu.sh [OPTIONS]

Sieve available GPU IDs (set \`gpu_id\'), and optionally set CUDA_VISIBLE_DEVICES.

OPTIONS:
	-h			Display this help message
	-b			Use (b)est-fitting mode. Default is (w)orst fit.
	-s			Set CUDA_VISIBLE_DEVICES
	-n INT		Acquire how many GPU/s. Default = -1 means all.
	-m INT		Minimal required memory of each GPU in MiB. Default = 0
	-i INT		Ignore these GPU/s. Can be set multiple times.

EXAMPLES:
	. find-gpu.sh
	. find-gpu.sh -n 2 -m 1500 -b -s -i 3 -i 4
EOF
}

_n_gpu_req=-1	# int = -1, how much GPU needed
_mem_lb=0		# int = 0, GPU memory lower-bound
_mode=w			# char = w, {b: Best fit, w: Worst fit}
_set=f			# bool = f, {t: set CUDA_VISIBLE_DEVICES, f: don't set}
_ignore=""		# int, +: ignore these GPU id

while getopts "hbsn:m:i:" opt; do
	case $opt in
	h)
		show_help
		return 0
		;;
	b)
		_mode=b
		;;
	s)
		_set=t
		;;
	n)
		_n_gpu_req=$OPTARG
		;;
	m)
		_mem_lb=$OPTARG
		;;
	i)
		_ignore="$_ignore $OPTARG"
		;;
	esac
done

unset gpu_id
n_gpu_found=0
if ! nvidia-smi &> /dev/null; then
	# `nvidia-smi` not usable -> GPU unavailable
	if [ ${_set} == "t" ]; then unset CUDA_VISIBLE_DEVICES; fi
	return 0
fi

# _i=0
if [ $_mode == "b" ]; then
	# _res=($(for _s in $_res; do echo $_i $_s && _i=`expr 1 + $_i`; done | \
	# 	sort -n -k 2))
	_res=(`nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sed "s/,//" | sort -nk 2`)
else
	# _res=($(for _s in $_res; do echo $_i $_s && _i=`expr 1 + $_i`; done | \
	# 	sort -n -k 2 -r))
	_res=(`nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sed "s/,//" | sort -nrk 2`)
fi

if [ ${_n_gpu_req} -lt 0 ]; then
	# echo using all GPUs
	_n_gpu_req=`expr ${#_res[@]} / 2`
	# echo ${n_gpu_req}
fi

for i in $(seq 0 2 `expr ${#_res[@]} - 1`); do
	_gid=${_res[i]}
	_mem=${_res[i+1]}

	_flag=0 # whether ignore this GPU
	for _ig in ${_ignore[@]}; do
		if [ $_ig -eq $_gid ]; then
			_flag=1
			break
		fi
	done
	if [ $_flag -eq 1 ]; then continue; fi

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
if [ ${_set} == "t" -a ${n_gpu_found} -gt 0 ]; then
	export CUDA_VISIBLE_DEVICES=${gpu_id}
# else
# 	unset CUDA_VISIBLE_DEVICES
fi

return 0
