#!/bin/bash

# shut this up, or this script will raise error and
# return when `ls` or `rm` fail in matching anything
# set -e

cd ..

clean()
{
    # print directory tree
    if [ $2 -gt 1 ]; then
        printf "|  %.0s" $(seq 2 $2)
    fi
    if [ $2 -gt 0 ]; then
        printf "|- "
    fi
    echo $1

    # clean log
    cd $1
    # tensorflow
    rm checkpoint 2>/dev/null
    rm SHDCH-*.data-*-of-* 2>/dev/null
    rm SHDCH-*.index 2>/dev/null
    rm SHDCH-*.meta 2>/dev/null
    rm events.out.tfevents.* 2>/dev/null
    # pytorch
    rm *.pth 2>/dev/null

    for d in `ls -d */ 2>/dev/null`; do
        clean `basename $d` `expr $2 + 1`
    done
    cd ..
    rmdir $1 2>/dev/null
}

for d in `ls -d */ | grep log`; do
    clean `basename $d` 0
done

