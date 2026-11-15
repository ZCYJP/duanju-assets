#!/bin/bash
# 在 h100-038 上多卡并行跑 50 个音色（Qwen3-TTS VoiceDesign）
# 每张卡一个进程分片不同角色；已存在的样音会跳过并补进报告
# OF=5 与卡数一致，每卡独占一个分片，显存充足
set -u
OF=5
GPUS="3 4 5 6 7"
PY=/ssd2/miniconda3/envs/videogen/bin/python
export PYTHONPATH=/ssd2/ttslib
export QWEN_TTS_DESIGN=/ssd2/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign
export CAST_JSON=/ssd2/duanju/characters/cast.json

mkdir -p /ssd2/duanju/voices/samples
PIDS=()
i=0
for G in $GPUS; do
  CUDA_VISIBLE_DEVICES=$G nohup $PY /ssd2/duanju/scripts/gen_voices_50.py \
    --shard "$i" --of "$OF" > "/tmp/voices50_shard${i}.log" 2>&1 &
  PIDS+=($!)
  echo "voice shard $i -> GPU $G pid ${PIDS[-1]}"
  i=$((i+1))
done

echo "all launched, waiting..."
for p in "${PIDS[@]}"; do wait "$p"; done
echo "=== ALL VOICE SHARDS DONE ==="
ls /ssd2/duanju/voices/samples/*.wav 2>/dev/null | wc -l
