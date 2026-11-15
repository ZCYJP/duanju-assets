#!/bin/bash
# 重生成仍含伪汉字的 4 张：强化 prompt 禁字 + 唯一 prefix + 完整性校验
set -u
cd /ssd2/duanju
SIDS="bg100_city_god_temple bg100_phone_booth bg100_republic_shanghai bg100_sword_sect_gate"
i=0
for sid in $SIDS; do
  port=$((8188 + i % 7))
  outdir="/ssd2/duanju/comfy_out_${port}"
  /usr/bin/rm -f "${outdir}/scene_b100/"*"${sid}"*.png "${outdir}/bg100_${sid}.png"
  /ssd2/miniconda3/envs/videogen/bin/python /ssd2/duanju/scripts/gen_bg100.py \
    --url "http://127.0.0.1:${port}" \
    --out "$outdir" \
    --scenes /ssd2/duanju/scripts/scenes_b100.json \
    --only "$sid" \
    --seed_offset 5000 \
    >> "/ssd2/duanju/logs/bg100/regen5_${sid}.log" 2>&1
  echo "done $sid (port $port)"
  i=$((i+1))
done
echo "=== regen5 done ==="
