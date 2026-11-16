#!/usr/bin/env bash
# 并行驱动 gen_jp200.py：每个 shard 一块卡，独立 cd/PY/URL/log。
# 用法:
#   run_jp200.sh <nshards>                      # 全量 200，按 index%nshards 分片
#   run_jp200.sh <nshards> --only sid1,sid2     # 只跑指定场景（此时所有 shard 都用同一 only，需自行拆分给不同 shard）
# 说明: --only 模式下建议用 run_only.sh，或用 --groups 传入每片各自的 only，逗号分隔的组用 ; 分隔。
set -u
N=${1:-7}
shift || true
OUT=/ssd2/duanju/scenes_jp200
PY=/ssd2/short_drama/.venv_tts/bin/python
SCRIPT=/ssd2/duanju/scripts/gen_jp200.py
mkdir -p "$OUT"
for ((i=0;i<N;i++)); do
  port=$((8188+i))
  ( cd "$OUT" && setsid "$PY" "$SCRIPT" --shard "$i" --of "$N" \
      --url "http://127.0.0.1:${port}" "$@" > "s${i}.log" 2>&1 < /dev/null ) &
done
wait
echo "all shards done"
