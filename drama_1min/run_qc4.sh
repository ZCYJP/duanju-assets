#!/bin/bash
cd /ssd2/duanju/drama_1min
/ssd2/short_drama/.venv_tts/bin/python qc_align_v4.py final/drama_1min_h3.mp4 final/timeline_h3.json > logs/qc_align_v4.log 2>&1
echo "EXIT=$?" >> logs/qc_align_v4.log
