!/bin/sh

ffmpeg -framerate 5 -pattern_type glob -i "out/sims/voronoi/imga*.png" \
-c:v libx264 -pix_fmt yuv420p out/sims/voronoi/outa.mp4
ffmpeg -framerate 5 -pattern_type glob -i "out/sims/voronoi/imgb*.png" \
-c:v libx264 -pix_fmt yuv420p out/sims/voronoi/outb.mp4