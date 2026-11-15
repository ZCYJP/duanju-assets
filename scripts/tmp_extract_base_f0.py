#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""提取每个角色的自然音高基线（从主音色样本）。

用途：情绪音色的 f0 目标必须用"相对基线倍率"表达，否则绝对区间会把
男声往女声区间拉（性别漂移）。本脚本从 /ssd2/duanju/voices/samples/*.wav
提取每个角色的基线 f0，写入 base_f0.json。

用法：python3 tmp_extract_base_f0.py
输出：/ssd2/duanju/voices/base_f0.json
"""
import glob
import json
import os
import re

import numpy as np
import soundfile as sf

SAMPLES = "/ssd2/duanju/voices/samples"
OUT = "/ssd2/duanju/voices/base_f0.json"


def f0_median(x, sr):
    frame = int(sr * 0.04)
    hop = int(sr * 0.02)
    lo, hi = int(sr / 400), int(sr / 70)
    vals = []
    for st in range(0, len(x) - frame, hop):
        seg = x[st:st + frame].astype(np.float64)
        if np.sqrt(np.mean(seg ** 2)) < 0.012:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, mode="full")[frame - 1:]
        if hi >= len(ac):
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        vals.append(sr / lag)
    return float(np.median(vals)) if vals else 0.0


def main():
    files = sorted(glob.glob(f"{SAMPLES}/*.wav"))
    print(f"[data] {len(files)} samples")
    base = {}
    for p in files:
        cid = os.path.basename(p).split("_")[0]
        try:
            w, sr = sf.read(p, dtype="float32")
            if w.ndim > 1:
                w = w[:, 0]
            f = f0_median(w, sr)
            base[cid] = round(f, 1)
        except Exception as ex:
            print(f"  [fail] {p}: {ex}")

    json.dump(base, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 按性别统计（从角色库取性别）
    cast = json.load(open("/ssd2/duanju/scripts/cast_all_150.json", encoding="utf-8"))
    g = {"男": [], "女": []}
    for c in cast:
        v = base.get(c["id"])
        if v:
            g[c.get("gender", "")].append(v)
    for k, v in g.items():
        if v:
            print(f"{k}: n={len(v)} 中位={np.median(v):.1f} "
                  f"范围=[{min(v):.1f}, {max(v):.1f}]")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
