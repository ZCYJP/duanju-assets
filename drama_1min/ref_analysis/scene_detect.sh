#!/bin/bash
set -u
OUT=/tmp/refan/scenes
mkdir -p "$OUT"
FF=/usr/local/bin/ffmpeg
echo "=== scene detect start $(date) ==="
# 全片场景切点，阈值 0.2；同时输出每帧 score 便于自适应阈值
$FF -v error -stats -i /tmp/ref.mp4 \
  -vf "select='gt(scene,0.2)',metadata=print:file=$OUT/cuts_t020.txt" \
  -an -f null - 2> "$OUT/scene_stat.log"
echo "=== scene detect done $(date) ==="
wc -l "$OUT/cuts_t020.txt"
