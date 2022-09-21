#!/bin/bash

# Fetch (specified types of files within) a remote folder recursively to local.
## Prerequisite
# Run `export-dir.sh` on source machine to get the tree structure (.sh) & file
#   list (.txt) of that source folder, and clone the folder tree at local in advance.
# SSH password-free login in destination machine should be configured in advance.
# `scp` is used for fetching.
## Arguments
# $1: path of (cloned) local destination folder to fetch
# $2: (optional) path to file list file (1 file per line)
#     If not provided, fetch all types of file with NO existence check,
#     multiprocessing & temporary file support.

IP=1.2.3.4
PORT=22
USER=itom
# $SRC_ROOT: path / parent folder where the remote folder to fetch lies
#   i.e. $SRC_ROOT/<folder-to-fetch> -> <folder-to-fetch>
SRC_ROOT=/home/itom/codes
N_PROCESSES=1

DEST=${1-"~/codes/ScanNet"}
if [ ! -d $DEST ]; then
	echo * NO SUCH FOLDER: $DEST
    echo Hint: use \`export-dir.sh\` to clone folder structure first
	exit
fi
cd $DEST
DEST=`pwd`
cd -

if [ ! -z $2 ]; then
	cd `dirname $2`
	FILE_LIST=`pwd`/`basename $2`  # use full path
	cd -
else
	unset FILE_LIST
fi

# temporary files
TMP_P=~/.cache/itom-fetch
if [ ! -d $TMP_P ]; then mkdir -p $TMP_P; fi
# create auxiliary fetching script
#	$1: full path of source file
#	$2: full path of destination file
echo 'if [ -f $2 ]; then' > ${TMP_P}/_fetch.sh # single '>'
echo '  echo skip: $2' >> ${TMP_P}/_fetch.sh
echo 'else' >> ${TMP_P}/_fetch.sh
echo '  echo fetch:' $USER@$IP:'$1 -\> $2' >> ${TMP_P}/_fetch.sh
echo '  scp -P' $PORT $USER@$IP:'$1 $2.tmp 2>/dev/null' >> ${TMP_P}/_fetch.sh
echo '  if [ $? -eq 0 -a ! -f $2 ]; then' >> ${TMP_P}/_fetch.sh
echo '	  mv $2.tmp $2' >> ${TMP_P}/_fetch.sh
echo '  elif [ -f $2.tmp ]; then' >> ${TMP_P}/_fetch.sh
echo '	  rm $2.tmp' >> ${TMP_P}/_fetch.sh
echo '  fi' >> ${TMP_P}/_fetch.sh
echo 'fi' >> ${TMP_P}/_fetch.sh

src=${SRC_ROOT}
process_id=0

dfs()
{
	cd $1
	src=$src/$1 # remote enter
	scp -P $PORT $USER@$IP:$src/* . 2> /dev/null
	for d in `ls -d */ 2>/dev/null`; do
		dfs `basename $d`
	done
	cd ..
	src=`dirname $src` # remote exit
}

loop_file()
{
	echo $1, $2
	for f in `cat $2`; do
		# f=$1/$f
		if [ $N_PROCESSES -gt 1 ]; then
			bash ${TMP_P}/_fetch.sh ${SRC_ROOT}/$f $1/$f &
			process_id=`expr 1 + $process_id`
			if [ $process_id -ge $N_PROCESSES ]; then
				wait
				process_id=0
			fi
		elif [ -f $1/$f ]; then
			echo skip: $1/$f
		else
			echo fetch: $USER@$IP:${SRC_ROOT}/$f -\> $1/$f
			scp -P $PORT $USER@$IP:${SRC_ROOT}/$f $1/$f.tmp
			# double-check of existence in case there is
			#   any other parallel download program
			if [ $? -eq 0 -a ! -f $1/$f ]; then
				mv $1/$f.tmp $1/$f
			elif [ -f $1/$f.tmp ]; then
				rm $1/$f.tmp
			fi
		fi
	done
}

cd $DEST/..
if [ -z $2 ]; then
	dest=`basename $DEST`
	dfs $dest
else
	dest_root=`pwd`
	loop_file ${dest_root} ${FILE_LIST}
fi

clear
echo Finish fetching $DEST
