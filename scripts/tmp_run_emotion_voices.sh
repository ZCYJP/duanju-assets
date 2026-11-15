#!/bin/bash
# 9 情绪音色：GPU3-7 五卡并行，1350 条（150 角色 x 9 情绪）
# ComfyUI 同时在跑，每卡已占 ~30GB，剩余 ~50GB 足够 Qwen3-TTS-1.7B
set -u
GPUS=(3 4 5 6 7)
for i in "${!GPUS[@]}"; do
  G="${GPUS[$i]}"
  rm -f "/tmp/emo_gpu${G}.log"
  CUDA_VISIBLE_DEVICES="$G" \
  PYTHONPATH=/ssd2/ttslib \
  HF_HOME=/ssd2/cache/huggingface \
  nohup /ssd2/miniconda3/envs/videogen/bin/python /ssd2/duanju/scripts/gen_emotion_voices.py \
    --shard "$i" --of 5 > "/tmp/emo_gpu${G}.log" 2>&1 &
  echo "emotion shard $i on GPU$G (pid $!)"
done
