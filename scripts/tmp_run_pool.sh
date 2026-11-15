#!/bin/bash
# 7 个 ComfyUI 实例共享任务池：8188(GPU0) 8189(GPU2) 8190-8194(GPU3-7)
set -u
POOL=/ssd2/duanju/pool
PORTS=(8188 8189 8190 8191 8192 8193 8194)
for i in "${!PORTS[@]}"; do
  P="${PORTS[$i]}"
  rm -f "/tmp/pool_w${i}.log"
  nohup /ssd2/miniconda3/envs/videogen/bin/python /ssd2/duanju/scripts/tmp_pool_worker.py \
    --url "http://127.0.0.1:${P}" --tasks "$POOL" --worker "w${i}_p${P}" \
    > "/tmp/pool_w${i}.out" 2>&1 &
  echo "worker $i -> port $P (pid $!)"
done
