#!/bin/bash
# 13 角色 x 26 变体 = 338 张时令变体重生成
# 注意：不用 set -u（bash 数组下标在 set -u 下会误报 unbound）
# 同时用固定 6 组（角色,端口）映射，避免数组越界
set -u
PAIRS=(
  "M10:8188" "M19:8189" "M41:8190"
  "M48:8191" "M59:8192" "M68:8193"
  "F29:8194" "F36:8188" "F44:8189"
  "F45:8190" "F51:8191" "F57:8192"
  "F37:8193"
)
for pair in "${PAIRS[@]}"; do
  cid="${pair%%:*}"
  port="${pair##*:}"
  LOG=/ssd2/duanju/characters/seasons_regen_${cid}.log
  setsid /usr/bin/python3 /ssd2/duanju/scripts/tmp_regen_seasons_for_ids.py \
    --ids "$cid" --url "http://127.0.0.1:${port}" > "$LOG" 2>&1 < /dev/null &
  echo "$cid -> :${port}"
  sleep 2
done
echo "all launched"
