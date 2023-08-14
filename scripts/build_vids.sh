#!/bin/sh

if [ "$#" -eq 2 ]; then
    ipath=$1
    opath=$2
    dooverwrite=-y
elif [ "$#" -eq 3 ]; then
    ipath=$1
    opath=$2
    dooverwrite=$3
fi


ffmpeg -framerate 5 -pattern_type glob -i "$ipath/imga*.png" \
-c:v libx264 -pix_fmt yuv420p $opath/outa.mp4 $dooverwrite

ffmpeg -framerate 5 -pattern_type glob -i "$ipath/imgb*.png" \
-c:v libx264 -pix_fmt yuv420p $opath/outb.mp4 $dooverwrite
