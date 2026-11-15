#!/bin/bash
cd /ssd2/duanju/drama_1min
/ssd2/miniconda3/envs/videogen/bin/python compose_h3_v2.py > logs/compose_v2.log 2>&1
echo "EXIT=$?" >> logs/compose_v2.log
