#!/bin/bash
# 重生成 VL 判定含模型自绘文字（含伪字）的图，seed_offset=2000
set -u
cd /ssd2/duanju
SIDS="bg100_city_god_temple bg100_crime_scene_flat bg100_phone_booth bg100_police_evidence bg100_rainy_neon bg100_republic_shanghai bg100_republic_station bg100_sword_sect_gate bg100_teleport_platform bg100_under_bridge"
# 按 7 个实例分片
i=0
for sid in $SIDS; do
  port=$((8188 + i % 7))
  outdir="/ssd2/duanju/comfy_out_${port}"
  setsid nohup /ssd2/miniconda3/envs/videogen/bin/python /ssd2/duanju/scripts/gen_bg100.py \
    --url "http://127.0.0.1:${port}" \
    --out "$outdir" \
    --scenes /ssd2/duanju/scripts/scenes_b100.json \
    --only "$sid" \
    --seed_offset 2000 \
    > "/ssd2/duanju/logs/bg100/regen2_${sid}.log" 2>&1 < /dev/null &
  echo "launch $sid -> port $port"
  i=$((i+1))
done
wait
echo "=== regen done ==="
