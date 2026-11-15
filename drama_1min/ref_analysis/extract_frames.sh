#!/bin/bash
set -u
OUT=/tmp/refan/frames
mkdir -p "$OUT"
FF=/usr/local/bin/ffmpeg
echo "=== extract start $(date) ==="
# 2fps 全片抽帧（用于景别/字幕样式/运镜分析）
$FF -v error -y -i /tmp/ref.mp4 -vf "fps=2,scale=360:640" -q:v 4 "$OUT/f_%05d.jpg"
# 另抽 0.5fps 大图（字幕细节 OCR 用）
mkdir -p /tmp/refan/frames_hi
$FF -v error -y -i /tmp/ref.mp4 -vf "fps=0.5" -q:v 2 "/tmp/refan/frames_hi/h_%05d.jpg"
echo "=== extract done $(date) ==="
ls /tmp/refan/frames | wc -l
ls /tmp/refan/frames_hi | wc -l
