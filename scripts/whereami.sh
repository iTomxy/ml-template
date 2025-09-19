#!/bin/bash

# Chech which processes I launch are using which GPUs

nvidia-smi pmon -s um -c 1 | awk 'NR > 2 {print $1" "$2" "$10}' | while read gid pid mem; do
	if [ "`whoami`" == "`ps -o user= -p $pid`" ]; then
		# gpu id, used memory, pid, command
		echo $gid $mem `ps --no-headers -o pid,cmd -p $pid`
	fi
done

