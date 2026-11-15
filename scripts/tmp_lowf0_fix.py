#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""超低音角色（base 70~95Hz）回正方案验证：分步降调 + 自适应窗长。

问题
----
v7 全量回正后 29 条未达标，全部是男性超低音角色：
    base 72~92Hz，r_before 2.5~5.0（严重漂移）
    需要 factor = target/r_before ≈ 0.22~0.37 的极端降调
    实测 r_after 偏高（1.09~2.85，预期 0.92~1.15）

两个假设
--------
H1. 【极端降调破坏低频谐波结构】
    factor=0.28 意味着 resample 到 sr/0.28 = 85.7kHz，
    再用 phase_vocoder 压回。这个极端操作会让基频区的
    谐波幅度严重失真，测量随之出错。
    对策：分步降调，每步 factor ∈ [0.50, 2.0]，
          总降调量不变但每步温和。

H2. 【分析窗太短，无法分辨低音基频与 2 倍频】
    f0=73Hz 时基周期 13.7ms，1024 样本(42.7ms) 只有 3.1 周期。
    实测 M03 在 w1024 下 ac 读 421Hz（误锁 5.6 倍），
    w4096 下读 87.3Hz（正确）。
    对策：按预期 f0 自适应窗长。

本脚本验证两者的组合效果。
"""
import glob
import json
import os

import numpy as np
import soundfile as sf
import librosa

VOICES = "/ssd2/duanju/voices"
FMIN, FMAX = 65.0, 800.0
TARGET_RATIO = {
    "xi": (1.00, 1.30), "nu": (0.95, 1.30), "ai": (0.82, 1.02),
    "le": (0.95, 1.12), "ju": (1.00, 1.30), "jing": (1.05, 1.40),
    "yan": (0.85, 1.05), "you": (0.90, 1.08), "xiu": (0.98, 1.22),
}


# ------------------------------------------------------------ 变调
def pitch_shift_once(x, sr, factor):
    if abs(factor - 1.0) < 0.005:
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


def pitch_shift_stepwise(x, sr, factor, lo=0.50, hi=2.0):
    """分步变调：把总 factor 拆成若干温和步骤。"""
    if abs(factor - 1.0) < 0.005:
        return x.astype(np.float32)
    steps = []
    f = factor
    while f < lo:
        steps.append(lo)
        f /= lo
    while f > hi:
        steps.append(hi)
        f /= hi
    steps.append(f)
    y = x.astype(np.float32)
    for s in steps:
        y = pitch_shift_once(y, sr, s)
    return y, len(steps)


# ------------------------------------------------------------ 估计器
def est_pyin_w(x, sr, frame=2048):
    try:
        f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=FMIN, fmax=FMAX,
                                sr=sr, frame_length=frame, hop_length=frame // 4)
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


def est_yin_thr_w(x, sr, frame=4096, threshold=0.15):
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


def est_ac_w(x, sr, frame=4096):
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


def measure_auto(x, sr, expect_f0=None):
    """自适应窗长测量：按预期 f0 选窗长。"""
    if expect_f0 is None or expect_f0 <= 0:
        frame = 2048
    elif expect_f0 < 110:
        frame = 4096
    elif expect_f0 < 200:
        frame = 2048
    else:
        frame = 1024
    v = [est_pyin_w(x, sr, frame),
         est_yin_thr_w(x, sr, max(frame, 2048)),
         est_ac_w(x, sr, frame)]
    v = [a for a in v if a > 0]
    if not v:
        return 0.0, frame, []
    return float(np.median(v)), frame, v


def main():
    rep = json.load(open(f"{VOICES}/pitch_fix_report_v7.json", encoding="utf-8"))
    bad = [r for r in rep["items"] if "r_after" in r and not r["in_range"]]

    def find_base(cid):
        for d in ("samples", "samples_ext"):
            g = sorted(glob.glob(f"{VOICES}/{d}/{cid}_*_voice.wav"))
            if g:
                return g[0]
        return None

    print("=" * 92)
    print("超低音角色回正验证：分步降调 vs 单次极端降调，自适应窗长测量")
    print("=" * 92)
    print(f"{'cid':<6}{'emo':<6}{'base':>7}{'r_bef':>8}{'target':>8}"
          f"{'单次r':>9}{'分步r':>9}{'步数':>6}{'窗长':>7}{'判定':>8}")
    n_single_ok = n_step_ok = 0
    rows = []
    for r in bad:
        cid, ek = r["cid"], r["emotion"]
        bp = find_base(cid)
        if not bp:
            continue
        xb, srb = sf.read(bp, dtype="float32")
        if xb.ndim > 1:
            xb = xb[:, 0]
        f0b, _, _ = measure_auto(xb, srb)
        if f0b <= 0:
            continue
        target_r = r["target_r"]
        lo_r, hi_r = TARGET_RATIO[ek]

        p = f"{VOICES}/emotions/{cid}/{cid}_{ek}.wav"
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        f0_0, _, _ = measure_auto(x, sr, expect_f0=None)
        if f0_0 <= 0:
            continue
        r_before = f0_0 / f0b
        fac = target_r / r_before

        # 单次极端降调
        y1 = pitch_shift_once(x, sr, fac)
        f1, fr1, _ = measure_auto(y1, sr, expect_f0=f0b * target_r)
        r1 = f1 / f0b if f0b else 0

        # 分步降调
        y2, nsteps = pitch_shift_stepwise(x, sr, fac)
        f2, fr2, _ = measure_auto(y2, sr, expect_f0=f0b * target_r)
        r2 = f2 / f0b if f0b else 0

        ok1 = (lo_r - 0.03) <= r1 <= (hi_r + 0.03)
        ok2 = (lo_r - 0.03) <= r2 <= (hi_r + 0.03)
        n_single_ok += int(ok1)
        n_step_ok += int(ok2)
        print(f"{cid:<6}{ek:<6}{f0b:7.1f}{r_before:8.3f}{target_r:8.3f}"
              f"{r1:9.3f}{r2:9.3f}{nsteps:6d}{fr2:7d}"
              f"{('OK' if ok2 else ('单步OK' if ok1 else 'FAIL')):>8}")
        rows.append((cid, ek, r1, r2, ok1, ok2))

    print(f"\n单次极端降调达标: {n_single_ok}/{len(rows)}")
    print(f"分步降调达标    : {n_step_ok}/{len(rows)}")


if __name__ == "__main__":
    main()
