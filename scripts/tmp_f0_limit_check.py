#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证：把 f0 搜索上限从 400Hz 放宽到 700Hz，是否影响基线的读数。

如果不影响（主音色 f0 都在 350Hz 以下，argmax 结果不变），
就可以放心用 700Hz 上限测量漂移音频，从而拿到可信的初始读数。
"""
import glob
import json
import os

import numpy as np
import soundfile as sf


def f0_ac(x, sr, fmax, octave_fix=False):
    frame = int(sr * 0.04)
    hop = int(sr * 0.02)
    lo, hi = int(sr / fmax), int(sr / 70)
    vals = []
    for st in range(0, len(x) - frame, hop):
        seg = x[st:st + frame].astype(np.float64)
        if np.sqrt(np.mean(seg ** 2)) < 0.012:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[frame - 1:]
        if hi >= len(ac):
            continue
        seg_ac = ac[lo:hi]
        if len(seg_ac) == 0:
            continue
        k = int(np.argmax(seg_ac))
        f = sr / (lo + k)
        if octave_fix:
            peak = seg_ac[k]
            hp = len(seg_ac) // 2
            hpk = seg_ac[hp] if hp < len(seg_ac) else 0.0
            if hpk > 0.85 * peak and f > 82.5:
                f /= 2.0
        vals.append(f)
    return float(np.median(vals)) if vals else 0.0


print("=== 基线样本：400Hz 上限 vs 700Hz 上限 ===")
files = sorted(glob.glob("/ssd2/duanju/voices/samples/*.wav"))
diff = []
for p in files:
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    a = f0_ac(x, sr, 400)
    b = f0_ac(x, sr, 700)
    diff.append((abs(b - a) / a if a else 0, os.path.basename(p), a, b))
diff.sort(reverse=True)
d = np.array([x[0] for x in diff])
print(f"样本数 {len(diff)}  相对差异: mean={d.mean()*100:.4f}%  "
      f"max={d.max()*100:.4f}%  |diff|>0.5% 的数量={np.sum(d>0.005)}")
for x in diff[:5]:
    print(f"  {x[1]:30s} 400hz={x[2]:7.1f}  700hz={x[3]:7.1f}  "
          f"差异 {x[0]*100:.3f}%")

# 与已存的 base_f0.json 比对
bm = json.load(open("/ssd2/duanju/voices/base_f0.json", encoding="utf-8"))
print("\n=== 与 base_f0.json 对比 ===")
diffs = []
for p in files:
    cid = os.path.basename(p).split("_")[0]
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    b = f0_ac(x, sr, 700)
    stored = bm.get(cid, 0)
    if stored > 0:
        diffs.append(abs(b - stored) / stored)
diffs = np.array(diffs)
print(f"  mean={diffs.mean()*100:.4f}%  max={diffs.max()*100:.4f}%  "
      f"|diff|>1% 的数量={np.sum(diffs>0.01)}")
