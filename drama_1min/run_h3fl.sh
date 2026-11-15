#!/usr/bin/env bash
cd /ssd2/duanju/drama_1min || exit 1
export PYTHONUNBUFFERED=1
python3 gen_h3_fl2va.py --ports 8192,8193,8191,8190,8188,8189,8194 --length 124
echo "EXIT=$?"
