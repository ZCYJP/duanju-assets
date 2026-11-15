#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""超低音倍频误锁的根因验证与修复：降低 FMIN + 加长窗。

根因推理
--------
v7 未达标案例的 r_after 全部【偏高】（预期 0.92~0.95，实测 1.10~2.15），
这是【倍频误锁】的典型特征（测到 2f 或 3f）。

为什么只在超低音男声出现？
    超低音男声 base f0 = 72~95Hz
    哀(yan) 目标 r = 0.95  ->  目标 f0 = 68~90Hz
    厌恶(ai) 目标 r = 0.92  ->  目标 f0 = 66~87Hz
    而 v7 的搜索下限 FMIN = 65Hz
    => 目标音高紧贴搜索下限，基频被压在搜索窗边缘，
       谐波能量在 2f 处更强 -> 估计器锁到 2f
    （f0=75Hz 时 2f=150Hz，150Hz 的谐波能量远高于 75Hz）

验证方案
--------
1. 构造「已知真实低 f0」的合成信号（含谐波，f0=70/80/90Hz），
   比较不同 FMIN / 窗长组合下的误锁率
2. 对真实超低音角色的目标音频，用新参数测量
3. 确认新参数不会破坏正常音域角色的精度

修复方案
--------
    FMIN: 65 -> 50 Hz
    frame: 低 f0 时用 8192（341ms，73Hz 有 25 个周期，足以分辨 2f）
"""
import glob
import json
import os

import numpy as np
import soundfile as sf
import librosa

VOICES = "/ssd2/duanju/voices"


def synth(f0_true, sr=24000, dur=3.0, n_harm=8, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(int(sr * dur)) / sr
    x = np.zeros_like(t)
    for h in range(1, n_harm + 1):
        if f0_true * h < sr / 2:
            x += (1.0 / h) * np.sin(2 * np.pi * f0_true * h * t
                                    + rng.uniform(0, 2 * np.pi))
    return (0.5 * x / np.max(np.abs(x))).astype(np.float32)


def est_pyin_w(x, sr, frame=2048, fmin=50.0, fmax=800.0):
    try:
        f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=fmin, fmax=fmax,
                                sr=sr, frame_length=frame, hop_length=frame // 4)
    except Exception:
        return 0.0
    v = f0[(f0 > fmin) & (f0 < fmax)]
    return float(np.median(v)) if len(v) else 0.0


def _yin_diff(seg, tau_max):
    n = len(seg)
    seg = seg - seg.mean()
    p0 = float(np.dot(seg, seg))
    acf = np.correlate(seg, seg, "full")[n - 1:]
    if len(acf) <= tau_max:
        return None
    sq = np.concatenate(([0.0], np.cumsum(seg ** 2)))
    ptau = sq[n:] - sq[:n]
    d = p0 + ptau[:tau_max + 1] - 2.0 * acf[:tau_max + 1]
    return np.maximum(d, 0.0)


def _parabolic(y, i):
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    if abs(den) < 1e-12:
        return float(i)
    d = 0.5 * (a - c) / den
    return float(i) + d if abs(d) <= 1.0 else float(i)


def est_yin_thr_w(x, sr, frame=4096, fmin=50.0, fmax=800.0, threshold=0.15):
    hop = frame // 4
    tau_min = max(int(sr / fmax), 2)
    tau_max = min(int(sr / fmin), frame - 1)
    if tau_max <= tau_min + 2:
        return 0.0
    vals = []
    for st in range(0, len(x) - frame, hop):
        seg = x[st:st + frame].astype(np.float64)
        if np.sqrt(np.mean(seg ** 2)) < 0.012:
            continue
        d = _yin_diff(seg, tau_max)
        if d is None:
            continue
        cum = np.cumsum(d[1:])
        taus = np.arange(1, len(d))
        dp = np.ones_like(d)
        dp[1:] = d[1:] * taus / np.maximum(cum, 1e-12)
        below = np.where(dp[tau_min:tau_max + 1] < threshold)[0]
        if len(below) == 0:
            i = int(np.argmin(dp[tau_min:tau_max + 1])) + tau_min
        else:
            i = int(below[0]) + tau_min
            while i + 1 <= tau_max and dp[i + 1] < dp[i]:
                i += 1
        tau = _parabolic(dp, i)
        if tau > 0:
            f = sr / tau
            if fmin <= f <= fmax:
                vals.append(f)
    return float(np.median(vals)) if vals else 0.0


def est_ac_w(x, sr, frame=4096, fmin=50.0, fmax=800.0):
    hop = frame // 4
    lo, hi = int(sr / fmax), int(sr / fmin)
    vals = []
    for st in range(0, len(x) - frame, hop):
        seg = x[st:st + frame].astype(np.float64)
        if np.sqrt(np.mean(seg ** 2)) < 0.012:
            continue
        seg = seg - seg.mean()
        acf = np.correlate(seg, seg, "full")[frame - 1:]
        if hi >= len(acf):
            continue
        s = acf[lo:hi]
        if len(s):
            vals.append(sr / (lo + int(np.argmax(s))))
    return float(np.median(vals)) if vals else 0.0


def measure(x, sr, fmin, frame):
    v = [est_pyin_w(x, sr, frame, fmin),
         est_yin_thr_w(x, sr, max(frame, 4096), fmin),
         est_ac_w(x, sr, frame, fmin)]
    v = [a for a in v if a > 0]
    return float(np.median(v)) if v else 0.0


def main():
    print("=" * 84)
    print("一、合成低 f0 信号：倍频误锁率（读数 / 真值）")
    print("=" * 84)
    cfgs = [
        ("FMIN65 w2048", 65.0, 2048),
        ("FMIN50 w4096", 50.0, 4096),
        ("FMIN50 w8192", 50.0, 8192),
        ("FMIN45 w8192", 45.0, 8192),
        ("FMIN50 w12288", 50.0, 12288),
    ]
    print(f"{'配置':<18}" + "".join(f"{('f0='+str(f)):>16}" for f in
                                     (70, 80, 90, 100, 120)))
    for name, fmin, frame in cfgs:
        row = f"{name:<18}"
        for ft in (70, 80, 90, 100, 120):
            got = []
            for seed in range(3):
                x = synth(ft, seed=seed)
                g = measure(x, 24000, fmin, frame)
                if g > 0:
                    got.append(g / ft)
            r = np.median(got) if got else 0
            lock = "锁" if r > 1.3 else ""
            row += f"{r:13.2f}{lock:>3}"
        print(row)

    print("\n" + "=" * 84)
    print("二、真实超低音角色主音色：新配置读数 vs v7 配置")
    print("=" * 84)
    rep = json.load(open(f"{VOICES}/pitch_fix_report_v7.json", encoding="utf-8"))
    bad_cids = sorted(set(r["cid"] for r in rep["items"]
                          if "r_after" in r and not r["in_range"]))

    def find_base(cid):
        for d in ("samples", "samples_ext"):
            g = sorted(glob.glob(f"{VOICES}/{d}/{cid}_*_voice.wav"))
            if g:
                return g[0]
        return None

    print(f"{'cid':<7}" + "".join(f"{n:>20}" for n, _, _ in cfgs))
    for cid in bad_cids:
        p = find_base(cid)
        if not p:
            continue
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        row = f"{cid:<7}"
        for name, fmin, frame in cfgs:
            row += f"{measure(x, sr, fmin, frame):20.1f}"
        print(row)

    print("\n" + "=" * 84)
    print("三、正常音域角色（抽查 8 个非失败角色）：新配置不能破坏精度")
    print("=" * 84)
    others = ["F01", "F17", "F02", "M01", "M07", "M25", "M40", "F30"]
    print(f"{'cid':<7}" + "".join(f"{n:>20}" for n, _, _ in cfgs))
    for cid in others:
        p = find_base(cid)
        if not p:
            continue
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        row = f"{cid:<7}"
        for name, fmin, frame in cfgs:
            row += f"{measure(x, sr, fmin, frame):20.1f}"
        print(row)


if __name__ == "__main__":
    main()
