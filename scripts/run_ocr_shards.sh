#!/usr/bin/env bash
# 分片并行跑 easyocr 伪字质检（在成图目录上）
set -u
PY=/ssd2/short_drama/.venv_tts/bin/python
SCRIPT=/ssd2/duanju/scripts/ocr_check.py
DIR=/ssd2/duanju/scenes_jp200_final
WORK=/ssd2/duanju/scenes_jp200
N=${1:-12}

cd "$DIR" || exit 1
/usr/bin/ls *.png | sort > "$WORK/_final_files.txt"
TOTAL=$(wc -l < "$WORK/_final_files.txt")
echo "total=$TOTAL shards=$N"

rm -f "$WORK"/ocr_shard*.log "$WORK"/ocr_shard*.json
split -n l/$N -d "$WORK/_final_files.txt" "$WORK/_chunk_"

i=0
for chunk in "$WORK"/_chunk_*; do
  imgs=$(tr '\n' ' ' < "$chunk")
  ( OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    setsid "$PY" "$SCRIPT" --imgs $imgs \
      --out "$WORK/ocr_shard${i}.json" --gpu cpu --langs ja,en \
      > "$WORK/ocr_shard${i}.log" 2>&1 < /dev/null ) &
  i=$((i+1))
done
wait
echo "ALL DONE $(date)"
