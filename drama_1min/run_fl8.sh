#!/usr/bin/env bash
cd /ssd2/duanju/drama_1min || exit 1
export PYTHONUNBUFFERED=1
sed -i 's/^STEPS = 6/STEPS = 8/' gen_h3_fl2va.py
sed -i 's#OUT = f"{BASE}/motion_h3_fl"#OUT = f"{BASE}/motion_h3_fl8"#' gen_h3_fl2va.py
sed -i 's#"h3fl/" + tag#"h3fl8/" + tag#' gen_h3_fl2va.py
python3 gen_h3_fl2va.py --ports 8192,8193,8191,8190,8188,8189,8194 --length 124
echo "EXIT=$?"
