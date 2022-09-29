#!/bin/bash

# Export the tree structure & file list of a folder to
#   a shell script & text file respectively for building
#   a same folder tree elsewhere & download the files.
## Arguments
# $1: path of folder to export
# $2: output script name
# $3: output file list text file name
## Inner Parameters
# FILE_TYPE: indicate what types of file to list

# FILE_TYPE=(scene*_*.sens scene*_*_2d-instance.zip)
FILE_TYPE=(' ')  # all

SRC=${1-"~/data/ScanNet"}
if [ ! -d $SRC ]; then
	echo * NO SUCH FOLDER: $SRC
	exit
fi
cd $SRC
SRC=`pwd`
cd - # back to last path

OUT_SHELL=${2-"copy-dir_`basename $SRC`.sh"}
cd `dirname $OUT_SHELL`
OUT_SHELL=`pwd`/`basename $OUT_SHELL`
cd -

OUT_TEXT=${3-"file-list_`basename $SRC`.txt"}
cd `dirname $OUT_TEXT`
OUT_TEXT=`pwd`/`basename $OUT_TEXT`
cd -

touch $OUT_SHELL  # permission test
if [ $? -ne 0 ]; then exit; fi
touch $OUT_TEXT   # permission test
if [ $? -ne 0 ]; then exit; fi
printf "\n\t%s\n\n" "$SRC  -->	$OUT_SHELL, $OUT_TEXT"

cd $SRC/..  # `/..` = `/`, no problem
src=`basename $SRC`

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

	echo "if [ ! -d $1 ]; then mkdir $1; fi" >> $OUT_SHELL
	cd $1
	echo "cd $1" >> $OUT_SHELL
	# src=$src/$1
	for ft in "${FILE_TYPE[@]}"; do  # enclosed by quotes
		# echo file type: $ft
		for f in `ls $ft 2>/dev/null`; do
			if [ -f $f ]; then	# is file
				echo $src/$f >> $OUT_TEXT
			fi
		done
	done
	for d in `ls -d */ 2>/dev/null`; do
		d=`basename $d`
		src=$src/$d
		dfs $d `expr 1 + $2`
		src=`dirname $src`
	done
	cd ..
	echo "cd .." >> $OUT_SHELL
	# src=`dirname $src`
}

if [ -f $OUT_TEXT ]; then
	# echo backing-up old file list: $OUT_TEXT
	# if [ -f $OUT_TEXT.bak ]; then
	# 	echo removing old backup: $OUT_TEXT.bak
	# 	rm $OUT_TEXT.bak
	# fi
	# mv $OUT_TEXT $OUT_TEXT.bak
	rm $OUT_TEXT
fi
echo "#/bin/bash" > $OUT_SHELL  # single '>'
dfs $src 0

