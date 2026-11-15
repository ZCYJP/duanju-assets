#!/bin/bash
# 重生成被 VL 判定"模型自绘文字"的背景图；旧图移入 quarantine 备查
set -u
BASE=/ssd2/duanju
IDS=(bg100_city_god_temple bg100_crime_scene_flat bg100_funeral_shop bg100_phone_booth
     bg100_police_evidence bg100_rainy_neon bg100_republic_shanghai bg100_republic_station
     bg100_song_tavern bg100_sword_sect_gate bg100_tang_market bg100_teleport_platform
     bg100_under_bridge)
OFFSET=${1:-1000}
QUAR=$BASE/scenes_b100_quarantine_$OFFSET
mkdir -p "$QUAR" "$BASE/logs/bg100"
for id in "${IDS[@]}"; do
  [ -f "$BASE/scenes_b100/$id.png" ] && mv "$BASE/scenes_b100/$id.png" "$QUAR/$id.png"
done
N=${#IDS[@]}
for ((p=0; p<7; p++)); do
  group=()
  for ((j=p; j<N; j+=7)); do group+=("${IDS[$j]}"); done
  csv=$(IFS=,; echo "${group[*]}")
  [ -z "$csv" ] && continue
  PORT=$((8188 + p))
  setsid /ssd2/short_drama/.venv_tts/bin/python "$BASE/scripts/gen_bg100.py" \
    --only "$csv" --url "http://127.0.0.1:${PORT}" --seed_offset "$((OFFSET * (p + 1)))" \
    > "$BASE/logs/bg100/regen_${OFFSET}_shard${p}.log" 2>&1 < /dev/null &
  echo "port ${PORT}: ${csv}"
done
sleep 2
pgrep -fc gen_bg100.py
