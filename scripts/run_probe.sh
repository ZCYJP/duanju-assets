#!/bin/bash
# 在 4 个 ComfyUI 实例上并行跑 4 个 probe，验证竖版出图链路
LOGDIR=/ssd2/duanju/logs/probe_b100
mkdir -p "$LOGDIR"
PORT=8188
for name in probe_city_sign probe_tea_sign probe_ancient_gate probe_susp_office; do
  setsid /ssd2/short_drama/.venv_tts/bin/python /ssd2/duanju/scripts/probe_gen.py \
    --url "http://127.0.0.1:${PORT}" --only "$name" \
    > "$LOGDIR/${name}.log" 2>&1 < /dev/null &
  echo "started ${name} on ${PORT}"
  PORT=$((PORT+1))
done
sleep 2
echo "--- running ---"
pgrep -fa probe_gen.py | head -10
