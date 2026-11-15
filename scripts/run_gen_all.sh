#!/bin/bash
# 7 个 ComfyUI 实例分片并行生成 100 张背景图
LOGDIR=/ssd2/duanju/logs/bg100
mkdir -p "$LOGDIR"
for i in 0 1 2 3 4 5 6; do
  PORT=$((8188 + i))
  setsid /ssd2/short_drama/.venv_tts/bin/python /ssd2/duanju/scripts/gen_bg100.py \
    --shard "$i" --of 7 --url "http://127.0.0.1:${PORT}" \
    > "$LOGDIR/shard${i}.log" 2>&1 < /dev/null &
  echo "shard ${i} -> port ${PORT}"
done
sleep 2
echo "--- running ---"
pgrep -fc gen_bg100.py
