#!/bin/bash
# 季节/节气变体任务池：7 个 ComfyUI 实例共享消费
set -u
POOL=/ssd2/duanju/pool_season
PORTS=(8188 8189 8190 8191 8192 8193 8194)
for i in "${!PORTS[@]}"; do
  P="${PORTS[$i]}"
  rm -f "/tmp/spool_w${i}.log"
  nohup /ssd2/miniconda3/envs/videogen/bin/python /ssd2/duanju/scripts/tmp_pool_worker2.py \
    --url "http://127.0.0.1:${P}" --tasks "$POOL" --worker "s${i}_p${P}" \
    > "/tmp/spool_w${i}.out" 2>&1 &
  echo "season worker $i -> port $P (pid $!)"
done
