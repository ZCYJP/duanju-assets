#!/bin/bash
set -u
mkdir -p /tmp/refan/pick
for t in 3 8 14 22 30 45 60 78 95 115 140 165 190 220 250 285 320 355 390 430 470 510 555 600 650 700 760 820 880 950 1020 1100 1180 1260 1340 1420 1500 1580 1620; do
  /usr/local/bin/ffmpeg -v error -ss "$t" -i /tmp/ref.mp4 -frames:v 1 -q:v 2 "/tmp/refan/pick/t_${t}.jpg" -y
done
ls /tmp/refan/pick | wc -l
