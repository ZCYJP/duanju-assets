#!/bin/bash
# 串行重生成 VL 判定含模型自绘文字的图（seed_offset=4000，prefix 已唯一化）
set -u
cd /ssd2/duanju
SIDS="bg100_crime_scene_flat bg100_phone_booth bg100_police_evidence bg100_rainy_neon bg100_republic_shanghai bg100_republic_station bg100_sword_sect_gate bg100_teleport_platform bg100_under_bridge"
i=1
for sid in $SIDS; do
  port=$((8188 + i % 7))
  outdir="/ssd2/duanju/comfy_out_${port}"
  /usr/bin/rm -f "${outdir}/scene_b100/"*"${sid}"*.png
  /ssd2/miniconda3/envs/videogen/bin/python /ssd2/duanju/scripts/gen_bg100.py \
    --url "http://127.0.0.1:${port}" \
    --out "$outdir" \
    --scenes /ssd2/duanju/scripts/scenes_b100.json \
    --only "$sid" \
    --seed_offset 4000 \
    >> "/ssd2/duanju/logs/bg100/regen4_${sid}.log" 2>&1
  echo "done $sid (port $port): $(grep -c ' -> ' /ssd2/duanju/logs/bg100/regen4_${sid}.log) ok"
  i=$((i+1))
done
echo "=== regen4 all done ==="
