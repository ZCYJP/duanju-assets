#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查基线口径一致性：base_f0_pyin（直接测）vs 校准测（升调后测）。

v5 失败案例 F17 暴露的疑点：
  - 情绪音频用【升调校准】测量（pyin(shift(x,1.6))/1.6）
  - 基线 base_f0_pyin.json 却是【直接 pyin】测量
  两把尺子不一致 -> 倍率全错。

本脚本对 150 条基线主音色同时算两种口径，并对比 F17 案例。
"""
import glob
import json
import os

import numpy as np
import soundfile as sf
import librosa

VOICES = "/ssd2/duanju/voices"
FMIN, FMAX = 70.0, 800.0
K = 1.6


def f0_pyin(x, sr):
    try:
        f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=FMIN, fmax=FMAX,
                                sr=sr, frame_length=1024, hop_length=256)
    except Exception:
        return 0.0
    v = f0[(f0 > FMIN) & (f0 < FMAX)]
    return float(np.median(v)) if len(v) else 0.0


def shift(x, sr, factor):
    if abs(factor - 1.0) < 1e-6:
        return x.astype(np.float32)
    n_in = len(x)
    y = librosa.resample(x.astype(np.float32), orig_sr=sr,
                         target_sr=max(int(sr / factor), 1000))
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    Ds = librosa.phase_vocoder(D, rate=1.0 / factor, hop_length=256, n_fft=1024)
    out = np.asarray(librosa.istft(Ds, hop_length=256, n_fft=1024), dtype=np.float32)
    if len(out) < n_in:
        out = np.pad(out, (0, n_in - len(out)))
    elif len(out) > n_in:
        out = out[:n_in]
    return out


def main():
    files = sorted(glob.glob(f"{VOICES}/samples/*_voice.wav") +
                   glob.glob(f"{VOICES}/samples_ext/*_voice.wav"))
    old = json.load(open(f"{VOICES}/base_f0_pyin.json", encoding="utf-8"))
    direct, calib = {}, {}
    for p in files:
        cid = os.path.basename(p).split("_")[0]
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        d = f0_pyin(x, sr)
        c = f0_pyin(shift(x, sr, K), sr) / K
        direct[cid] = round(d, 1)
        calib[cid] = round(c, 1)
        print(f"  {cid}  direct={d:7.1f}  calib={c:7.1f}  "
              f"diff={(c-d)/d*100:+6.1f}%", flush=True)

    json.dump(calib, open(f"{VOICES}/base_f0_calib.json", "w",
                           encoding="utf-8"), indent=2)

    arr_d = np.array([direct[c] for c in direct])
    arr_c = np.array([calib[c] for c in calib])
    rel = (arr_c - arr_d) / arr_d * 100
    print("\n=== 两种口径对比（150 条基线主音色）===")
    print(f"direct  均值 {arr_d.mean():7.1f}  中位 {np.median(arr_d):7.1f}")
    print(f"calib   均值 {arr_c.mean():7.1f}  中位 {np.median(arr_c):7.1f}")
    print(f"相对差  中位 {np.median(rel):+6.1f}%  p90 {np.percentile(np.abs(rel),90):6.1f}%"
          f"  最大 {np.max(np.abs(rel)):6.1f}%")
    big = [(c, direct[c], calib[c], (calib[c]-direct[c])/direct[c]*100)
           for c in direct if abs((calib[c]-direct[c])/direct[c]*100) > 10]
    print(f"\n差异 >10% 的角色（{len(big)} 个）：")
    for c, d, cc, r in sorted(big, key=lambda t: -abs(t[3])):
        print(f"  {c}: direct={d:7.1f}  calib={cc:7.1f}  {r:+6.1f}%")


if __name__ == "__main__":
    main()
