#!/bin/bash
cd /ssd2/duanju/drama_1min/sample_15s
export REGEN=1
/ssd2/miniconda3/envs/videogen/bin/python -u gen_frames_15s.py < /dev/null > /ssd2/duanju/drama_1min/sample_15s/logs/frames.log 2>&1
echo "FRAMES_DONE rc=$?" >> /ssd2/duanju/drama_1min/sample_15s/logs/frames.log
