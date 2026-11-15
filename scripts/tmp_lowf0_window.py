#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""超低音角色的 f0 估计：分析窗长度的影响。

问题
----
v7 全量回正 29 条未达标，**全部是男性超低音角色**（base 72~92Hz），
且三算法分歧度极高（spread 中位 146%）。

根因假设：分析窗太短。
    frame=1024 @ 24kHz = 42.7ms
    对 f0=73Hz，基周期 = 13.7ms -> 窗内只有 3.1 个周期
    自相关/自研 YIN 在 3 个周期上无法可靠区分
    基频与 2 倍频（周期 6.8ms，窗内 6.2 个）-> 必然误锁

对策：低 f0 角色用更长分析窗。
    frame=2048 -> 85ms（6.2 周期）
    frame=4096 -> 171ms（12.4 周期）

本脚本验证：
  1. 各窗长下，超低音角色主音色的 f0 读数一致性
  2. 往返误差（低基线角色 + 大降调）
  3. 与「谐波求和」交叉的效果
"""
import glob
import json
import os

import numpy as np
import soundfile as sf
import librosa

VOICES = "/ssd2/duanju/voices"
FMIN, FMAX = 65.0, 800.0


def shift(x, sr, factor):
    if abs(factor - 1.0) < 1e-6:
        return x.astype(np.float32)
    n_in = len(x)
    rms_in = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    y = librosa.resample(x.astype(np.float32), orig_sr=sr,
                         target_sr=max(int(sr / factor), 1000))
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    Ds = librosa.phase_vocoder(D, rate=1.0 / factor, hop_length=256, n_fft=1024)
    out = np.asarray(librosa.istft(Ds, hop_length=256, n_fft=1024), dtype=np.float32)
    if len(out) < n_in:
        out = np.pad(out, (0, n_in - len(out)))
    elif len(out) > n_in:
        out = out[:n_in]
    rms_out = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    if rms_out > 1e-8 and rms_in > 1e-8:
        out = out * (rms_in / rms_out)
    pk = float(np.max(np.abs(out)))
    if pk > 0.99:
        out = out * (0.99 / pk)
    return out


def est_pyin_w(x, sr, frame=1024):
    try:
        f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=FMIN, fmax=FMAX,
                                sr=sr, frame_length=frame,
                                hop_length=frame // 4)
    except Exception:
        return 0.0
    v = f0[(f0 > FMIN) & (f0 < FMAX)]
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


def est_yin_thr_w(x, sr, frame=2048, threshold=0.15):
    hop = frame // 4
    tau_min = max(int(sr / FMAX), 2)
    tau_max = min(int(sr / FMIN), frame - 1)
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
            if FMIN <= f <= FMAX:
                vals.append(f)
    return float(np.median(vals)) if vals else 0.0


def est_ac_w(x, sr, frame=2048):
    hop = frame // 4
    lo, hi = int(sr / FMAX), int(sr / FMIN)
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


CONFIGS = {
    "w1024": lambda x, sr: (est_pyin_w(x, sr, 1024),
                             est_yin_thr_w(x, sr, 1024),
                             est_ac_w(x, sr, 1024)),
    "w2048": lambda x, sr: (est_pyin_w(x, sr, 2048),
                             est_yin_thr_w(x, sr, 2048),
                             est_ac_w(x, sr, 2048)),
    "w4096": lambda x, sr: (est_pyin_w(x, sr, 4096),
                             est_yin_thr_w(x, sr, 4096),
                             est_ac_w(x, sr, 4096)),
}


def med3(vals):
    v = [x for x in vals if x > 0]
    return float(np.median(v)) if v else 0.0


def main():
    rep = json.load(open(f"{VOICES}/pitch_fix_report_v7.json", encoding="utf-8"))
    bad = [r for r in rep["items"] if "r_after" in r and not r["in_range"]]
    bad_cids = sorted(set(r["cid"] for r in bad))
    print(f"未达标角色 {len(bad_cids)} 个: {bad_cids}\n")

    # 找这些角色的基线主音色
    def find_base(cid):
        for d in ("samples", "samples_ext"):
            g = sorted(glob.glob(f"{VOICES}/{d}/{cid}_*_voice.wav"))
            if g:
                return g[0]
        return None

    print("=" * 88)
    print("一、超低音角色基线主音色：三种窗长下的读数（应彼此接近，且接近真实值）")
    print("=" * 88)
    print(f"{'cid':<7}" + "".join(f"{k:>26}" for k in CONFIGS))
    print(f"{'':<7}" + "".join(f"{'pyin/yin/ac (spread%)':>26}" for _ in CONFIGS))
    for cid in bad_cids:
        p = find_base(cid)
        if not p:
            continue
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        row = f"{cid:<7}"
        for name, fn in CONFIGS.items():
            v = fn(x, sr)
            if all(u > 0 for u in v):
                sp = (max(v) - min(v)) / np.median(v) * 100
                row += f"  {v[0]:6.1f}/{v[1]:6.1f}/{v[2]:6.1f} ({sp:4.0f}%)"
            else:
                row += f"  {v[0]:6.1f}/{v[1]:6.1f}/{v[2]:6.1f}      "
        print(row)

    print("\n" + "=" * 88)
    print("二、往返误差：低基线角色，降调到 0.35~0.5（模拟严重漂移场景）")
    print("=" * 88)
    print(f"{'cfg':<8}" + "".join(f"{('x'+str(f)):>12}" for f in (0.5, 0.4, 0.35))
          + f"{'中位':>10}{'p90':>10}")
    for name, fn in CONFIGS.items():
        errs = []
        cells = ""
        for fac in (0.5, 0.4, 0.35):
            a = []
            for cid in bad_cids[:8]:
                p = find_base(cid)
                if not p:
                    continue
                x, sr = sf.read(p, dtype="float32")
                if x.ndim > 1:
                    x = x[:, 0]
                y = shift(x, sr, fac)
                f0m = med3(fn(y, sr))
                f0b = med3(fn(x, sr))
                if f0m > 0 and f0b > 0:
                    a.append(abs(f0m / f0b - fac) / fac * 100)
            errs += a
            cells += f"{(np.median(a) if a else -1):11.1f}%"
        e = np.array(errs)
        print(f"{name:<8}{cells}{np.median(e):9.1f}%{np.percentile(e,90):9.1f}%")


if __name__ == "__main__":
    main()
