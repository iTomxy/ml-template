#!/bin/bash

# Send (specified types of files within) a folder recursively to another machine.
## Prerequisites
# SSH password-free login in destination machine should be configured in advance.
# `scp` is used for sending.
## Arguments
# $1: path of folder to send

IP=1.2.3.4
PORT=22
USER=itom
# $DEST_ROOT: path / parent folder where the folder to send
#   will be placed in the destination machine,
#   i.e. <folder-to-send> -> $DEST_ROOT/<folder-to-send>
DEST_ROOT=/home/itom
N_PROCESSES=1
# FILE_TYPE=(scene*_*.sens scene*_*_2d-instance.zip)
FILE_TYPE=(' ')  # all

SRC=${1-"~/data/ScanNet"}
if [ ! -d $SRC ]; then
	echo * NO SUCH FOLDER: $SRC
	exit
fi
cd $SRC
SRC=`pwd` # full path -> begins with '/'

# temporary files
TMP_P=~/.cache/itom-send$SRC # NO '/' cuz $SRC begins with one
SENT_LOG=${TMP_P}/sent.txt
if [ ! -d $TMP_P ]; then mkdir -p $TMP_P; fi
touch $SENT_LOG
gather_logs()
{
	# gather (potential) ungathered sent logs
	for log_f in `ls ${TMP_P}/sent-*.txt 2>/dev/null`; do
		echo gather: $log_f
		cat ${log_f} >> $SENT_LOG
		rm ${log_f}
	done
}
# create auxiliary sending script
#	$1: full path of source file
#	$2: full path of destination file
#	$3: process ID
echo 'scp -P' $PORT '$1' $USER@$IP':$2' > ${TMP_P}/_send.sh # single '>'
echo 'if [ $? -eq 0 ]; then' >> ${TMP_P}/_send.sh
echo '	pid=${3-"0"}' >> ${TMP_P}/_send.sh
echo '	echo $1 >' ${TMP_P}'/sent-$pid.txt' >> ${TMP_P}/_send.sh
echo 'fi' >> ${TMP_P}/_send.sh

dest=${DEST_ROOT}
process_id=0

send()
{
	r=`grep "$1" ${SENT_LOG}`
	if [ "$r" != "" ]; then  # already sent
		echo skip: $1
	else
		dest_f=$dest/`basename $1`
		echo $1 -\> $dest_f

		if [ $N_PROCESSES -lt 2 ]; then
			scp -P $PORT $1 $USER@$IP:$dest_f
			echo $1 >> $SENT_LOG
		else
			bash ${TMP_P}/_send.sh $1 $dest_f $process_id &
			process_id=`expr 1 + $process_id`
			if [ $process_id -ge $N_PROCESSES ]; then
				wait
				process_id=0 # reset
				gather_logs
			fi
		fi
	fi
}

dfs()
{
	cd $1
	dest=$dest/$1 # remote enter
	for ft in "${FILE_TYPE[@]}"; do  # enclosed by quotes
		# echo file type: $ft
		for src_f in `ls $ft 2>/dev/null`; do
			if [ -f $src_f ]; then	# is file
				# echo $src_f
				send `pwd`/`basename $src_f`  # use full path
			fi
		done
	done
	for d in `ls -d */ 2>/dev/null`; do
		dfs `basename $d`
	done
	cd ..
	dest=`dirname $dest` # remote exit
}

gather_logs
cd $SRC/..  # `/..` = `/`, no problem
src=`basename $SRC`
dfs $src
if [ $N_PROCESSES -gt 1 ]; then
	gather_logs
fi

clear
echo Finish sending $SRC
