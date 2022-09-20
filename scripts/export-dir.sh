#!/bin/bash

# Export the structure of a folder to a shell script
#	for building a same folder tree elsewhere.
## Arguments
# $1: path of folder to export
# $2: output script name

SRC=${1-"~/data/ScanNet"}
if [ ! -d $SRC ]; then
	echo * NO SUCH FOLDER: $SRC
	exit
elif [ $SRC == "." ]; then
	SRC=`pwd`
elif [ $SRC == ".." ]; then
	cd ..
	SRC=`pwd`
	cd -  # back to last dir
fi

OUT=${2-"copy-dir_`basename $SRC`.sh"}
if [ `dirname $OUT` == "." ]; then
	OUT=`pwd`/`basename $OUT`
fi

touch $OUT	# permission test
if [ $? -ne 0 ]; then exit; fi
printf "\n\t%s\n\n" "$SRC  -->	$OUT"

dfs()
{
	# print folder structure
	if [ $2 -gt 1 ]; then
		printf "|  %.0s" $(seq 2 $2)
	fi
	if [ $2 -gt 0 ]; then
		printf "|- "
	fi
	echo $1/

	echo "if [ ! -d $1 ]; then mkdir $1; fi" >> $OUT
	cd $1
	echo "cd $1" >> $OUT
	for d in `ls -d */ 2>/dev/null`; do
		dfs `basename $d` `expr 1 + $2`
	done
	cd ..
	echo "cd .." >> $OUT
}

src=`basename $SRC`
echo "#/bin/bash" > $OUT
dfs $src 0

