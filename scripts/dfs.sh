#!/bin/bash
# Traverse a folder recursively to do something,
# e.g. delte some files,
# and print the folder structure by the way.

# set -e
root=$1
rm_empty_dir=${2:-0} # whether to delete empty folder by the way


doit()
{
    # do something in the current folder, e.g. clean log
    rm events.out.tfevents.* 2>/dev/null
    rm *.pth 2>/dev/null
}


dfs()
{
    # print directory tree
    if [ $2 -gt 1 ]; then
        printf "|  %.0s" $(seq 2 $2)
    fi
    if [ $2 -gt 0 ]; then
        printf "|- "
    fi
    echo $1

    cd $1
    doit
    for d in `ls -d */ 2>/dev/null`; do
        dfs ${d%/} `expr $2 + 1`
    done
    cd ..
    if [ $rm_empty_dir -ne 0 ]; then
        rmdir $1 2>/dev/null # NO -f & -r, only delete EMPTY folder
    fi
}


root=`realpath $root`
if [ -d $root ]; then
    cwd=`pwd`
    cd `dirname $root`
    dfs `basename $root` 0
    cd $cwd
fi
