#!/bin/bash

dset=$1
dset=${dset:="wikipedia"}
# echo $dset

if [ ! -d $dset ]; then
    mkdir $dset
fi
cd $dset

case $dset in
wikipedia)
    # SRC_P=/home/dataset/wikipedia_dataset
    SRC_P=/home/tom/dataset/wikipedia
    FILE=(images.vgg19.mat texts.wiki.lda.10.mat labels.wiki.mat)
    for f in ${FILE[@]}; do
        # echo $f
        ln -s $SRC_P/$f
    done
    ;;
flickr25k)
    SRC_P=/home/tom/dataset/flickr
    FILE=(images.flickr25k.vgg19.4096d.mat texts.npy labels.24.npy clean_id.npy)
    for f in ${FILE[@]}; do
        # echo $f
        ln -s $SRC_P/$f
    done
    ;;
nuswide-tc10)
    SRC_P=/home/tom/dataset/nuswide
    FILE=(images.nuswide.vgg19.4096d.h5 texts.AllTags1k.npy labels.tc-10.npy clean_id.tc10.npy)
    for f in ${FILE[@]}; do
        # echo $f
        ln -s $SRC_P/$f
    done
    ;;
*)
    echo Not implemented: $dset
    exit
    ;;
esac

