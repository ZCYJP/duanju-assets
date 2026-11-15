#!/bin/bash
# 100 个新角色音色：GPU3-7 五卡并行，每卡一个进程
# ComfyUI 同时在跑，每卡已占 ~30GB，剩余 ~50GB 足够 Qwen3-TTS-1.7B
set -u
GPUS=(3 4 5 6 7)
for i in "${!GPUS[@]}"; do
  G="${GPUS[$i]}"
  CUDA_VISIBLE_DEVICES="$G" \
  PYTHONPATH=/ssd2/ttslib \
  HF_HOME=/ssd2/cache/huggingface \
  nohup /ssd2/miniconda3/envs/videogen/bin/python /ssd2/duanju/scripts/gen_voices_100.py \
    --shard "$i" --of 5 > "/tmp/voices100_gpu${G}.log" 2>&1 &
  echo "voice shard $i on GPU$G (pid $!)"
done
