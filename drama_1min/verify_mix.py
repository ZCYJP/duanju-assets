#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 mix_audio：逐镜检查人声落点是否精确等于 PAD_HEAD"""
import os, sys, json
sys.path.insert(0, "/ssd2/duanju/drama_1min")
import numpy as np
import mix_audio
from shots_1min import SHOTS

BASE = "/ssd2/duanju/drama_1min"
VOICE_DIR = f"{BASE}/voice"
MOTION = f"{BASE}/motion_h3_fln"
SR = mix_audio.SR

shots = []
for s in SHOTS:
    vp = None
    for sp, tx in s["dialogue"]:
        c = f"{VOICE_DIR}/{s['name']}_{sp}.wav"
        if os.path.exists(c):
            vp = c
            break
    is_last = (s["id"] == len(SHOTS))
    shots.append({"name": s["name"], "voice_path": vp,
                  "seg_dur": (mix_audio and (124/24 if is_last else 118/24))})

print("=== 逐镜混合 + 落点验证 ===")
allok = True
for s in shots:
    mixed, meta = mix_audio.mix_shot(f"{MOTION}/{s['name']}.mp4",
                                     s["voice_path"], s["seg_dur"], 0.30)
    mono = mixed.mean(axis=1)
    # 从成品混音里反查真实发声点
    env, win = mix_audio.envelope(mono)
    if env.size == 0:
        continue
    peak = float(env.max())
    thr = peak * 0.15
    st = None
    for i in range(env.size):
        if i + 2 < env.size and env[i] >= thr and env[i+1] >= thr and env[i+2] >= thr:
            st = i
            break
    if st is None:
        st = int(np.argmax(env))
    detected = st * win / SR
    # 人声峰值是否达标（人声段 RMS 应接近 -20dB，环境音 -34dB）
    vseg = mono[int(0.30*SR):int(min(0.30+1.0, s['seg_dur'])*SR)]
    vdb = mix_audio.rms_db(vseg)
    ok = abs(detected - 0.30) < 0.12
    allok = allok and ok
    print(f"  [{'OK ' if ok else 'BAD'}] {s['name']:<20} "
          f"期望0.30s 实测{detected:.3f}s  人声段{vdb:.1f}dBFS  "
          f"原声起点{meta.get('raw_speech_start')}s 增益{meta.get('voice_gain_db')}dB")

print(f"\n全部落点合规: {allok}")
