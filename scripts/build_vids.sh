!/bin/sh

ffmpeg -framerate 5 -pattern_type glob -i "out/sims/$1/imga*.png" \
-c:v libx264 -pix_fmt yuv420p out/sims/$1/outa.mp4
ffmpeg -framerate 5 -pattern_type glob -i "out/sims/$1/imgb*.png" \
-c:v libx264 -pix_fmt yuv420p out/sims/$1/outb.mp4