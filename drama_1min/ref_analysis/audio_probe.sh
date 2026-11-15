#!/bin/bash
set -u
OUT=/tmp/refan
mkdir -p "$OUT"
FF=/usr/local/bin/ffmpeg
FP=/usr/local/bin/ffprobe
echo "=== audio probe $(date) ==="
$FP -v error -show_streams -select_streams a:0 -of default=noprint_wrappers=1 /tmp/ref.mp4 > "$OUT/a_stream.txt"
# 整片 ebur128 响度
$FF -v error -nostats -i /tmp/ref.mp4 -af ebur128=peak=true -f null - 2> "$OUT/ebur128.log"
# 整体 astats
$FF -v error -nostats -i /tmp/ref.mp4 -af astats=metadata=1:reset=0 -f null - 2> "$OUT/astats.log"
# 抽出 16k 单声道 wav 供 python 分析
$FF -v error -y -i /tmp/ref.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le "$OUT/ref16k.wav"
echo "=== audio probe done $(date) ==="
ls -l "$OUT/ref16k.wav"
