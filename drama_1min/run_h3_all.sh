#!/usr/bin/env bash
# H3 全量 12 镜批量生成（后台）
cd /ssd2/duanju/drama_1min || exit 1
export PYTHONUNBUFFERED=1
python3 gen_h3.py --ports 8192,8193,8191,8190,8188,8189,8194 --length 124 --ref-size match
echo "EXIT=$?"
