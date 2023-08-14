#!/bin/sh


if [ "$#" -eq 1 ]; then
    fpath=$1
    dooverwrite=-y
elif [ "$#" -eq 2 ]; then
    fpath=$1
    dooverwrite=$2
fi


ffmpeg -framerate 5 -pattern_type glob -i "$fpath/imga*.png" \
-c:v libx264 -pix_fmt yuv420p $fpath/outa.mp4 $dooverwrite

ffmpeg -framerate 5 -pattern_type glob -i "$fpath/imgb*.png" \
-c:v libx264 -pix_fmt yuv420p $fpath/outb.mp4 $dooverwrite
