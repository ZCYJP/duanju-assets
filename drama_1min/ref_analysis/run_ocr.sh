#!/bin/bash
cd /tmp/refan
export OMP_NUM_THREADS=8
/ssd2/miniconda3/envs/videogen/bin/python /tmp/refan/subtitle_style.py < /dev/null > /tmp/refan/subtitle_style.log 2>&1
echo "OCR_DONE rc=$?" >> /tmp/refan/subtitle_style.log
