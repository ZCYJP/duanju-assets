#!/bin/bash
# 12 个性别不符角色重生成（3 批 x 4 角色并行）
# GPU 绑定: 8188->0(已占30G) 8189->2 8190->3 8191->4 8192->5 8193->6 8194->7
# GPU1 完全空闲 -> VL 模型放 GPU1
# ComfyUI 端口与 VL 卡错开：只用 8189-8194 (GPU2-7)
set -u
BATCHES=(
  "M10 M19 M41 M48"
  "M59 M68 F29 F36"
  "F44 F45 F51 F57"
)
PORTMAP=("8189 8190 8191 8192" "8193 8194 8189 8190" "8191 8192 8193 8194")
i=0
for ids in "${BATCHES[@]}"; do
  LOG=/ssd2/duanju/characters/regen_batch${i}.log
  {
    echo "=== batch $i ids=$ids VL_GPU=1 ==="
    k=0
    read -r -a PM <<< "${PORTMAP[$i]}"
    for cid in $ids; do
      PORT=${PM[$k]}
      echo "--- $cid via ComfyUI :$PORT ---"
      CUDA_VISIBLE_DEVICES=1 \
        /ssd2/envs/qwentts/bin/python /ssd2/duanju/scripts/tmp_regen_gender_mismatch.py \
        --ids $cid --cand 4 --url http://127.0.0.1:$PORT \
        --log /ssd2/duanju/characters/gender_regen_log_batch${i}.json 2>&1 \
        | grep -vE "^Loading checkpoint|it/s\]|torch_dtype|fast processor"
      k=$((k+1))
    done
  } > $LOG 2>&1 &
  i=$((i+1))
done
echo "3 batches launched, VL on GPU1"
