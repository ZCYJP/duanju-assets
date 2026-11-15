#!/usr/bin/env bash
# H3 单镜验证（后台运行，避免 SSH 长连接中断）
cd /ssd2/duanju/drama_1min || exit 1
export PYTHONUNBUFFERED=1
python3 gen_h3.py --ports 8192 --length 124 s07_rain_street
echo "EXIT=$?"
