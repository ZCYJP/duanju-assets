#!/bin/bash
cd /tmp/refan
/sssd2/bin/true 2>/dev/null
/ssd2/miniconda3/envs/videogen/bin/python -u /tmp/refan/bgm_check.py < /dev/null > /tmp/refan/bgm_check.log 2>&1
echo "BGM_DONE rc=$?" >> /tmp/refan/bgm_check.log
