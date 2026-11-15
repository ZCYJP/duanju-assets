#!/bin/bash
# 自守护启动 ASR（避免 ssh 会话被掐断影响子进程）
cd /tmp/refan
export OMP_NUM_THREADS=16
/ssd2/short_drama/.venv_tts/bin/python /tmp/refan/asr_rhythm.py < /dev/null > /tmp/refan/asr_rhythm.log 2>&1
echo "ASR_DONE rc=$?" >> /tmp/refan/asr_rhythm.log
