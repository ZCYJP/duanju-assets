#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""寻找对真实语音稳定、对谐波误锁鲁棒的 f0 测量方案。

诊断结论
--------
变调算法本身没问题：合成信号（220Hz 正弦）升调偏差 <0.4%，完全对称。
偏差全部来自 f0【测量】：
    F02 连续降调时实测 f0 在 220.2 / 275.9 / 292.7 / 303.8 之间跳，
    而理论值应单调下降 257->257->240->229->200。
    这些跳变值恰好是基频的谐波间隔（214.3 x 1 / 1.28 / 1.36 / 1.42），
    说明自相关在每个变调后的音频上锁到了不同的谐波。

候选方案
--------
1. ac            自相关（原方案，不稳定）
2. ac_oct        自相关 + 倍频折叠
3. yin           librosa.yin
4. pyin          librosa.pyin（概率 YIN，对谐波误锁最鲁棒）
5. hpss+pyin     先 HPSS 分出谐波成分再 pyin
6. autocorr 多窗口投票  对多个 frame 长度取众数

评估标准
--------
对每条音频，用【多种子重复】的方式构造已知音高的测试信号：
  从真实语音的 f0 轨迹估计基频，合成同音高的正弦+谐波信号，
  测量各算法的误差。
同时在真实音频上比较各算法给出的 f0 分布合理性。
"""
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
import librosa

EMOTIONS = "/ssd2/duanju/voices/emotions"
BASE_F0 = "/ssd2/duanju/voices/base_f0.json"
SR = 24000


def ac(x, sr, fmin=70, fmax=400, octave_fix=False):
    frame = int(sr * 0.04)
    hop = int(sr * 0.02)
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
        if len(s) == 0:
            continue
        k = int(np.argmax(s))
        f = sr / (lo + k)
        if octave_fix:
            peak = s[k]
            hp = len(s) // 2
            hpk = s[hp] if hp < len(s) else 0.0
            if hpk > 0.85 * peak and f > 82.5:
                f /= 2.0
        vals.append(f)
    return float(np.median(vals)) if vals else 0.0


def yin(x, sr, fmin=70, fmax=400):
    try:
        f0 = librosa.yin(x.astype(np.float32), fmin=fmin, fmax=fmax, sr=sr,
                         frame_length=1024, hop_length=256)
    except Exception:
        return 0.0
    v = f0[(f0 > fmin) & (f0 < fmax)]
    return float(np.median(v)) if len(v) else 0.0


def pyin(x, sr, fmin=70, fmax=400):
    try:
        f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=fmin, fmax=fmax,
                                sr=sr, frame_length=1024, hop_length=256)
    except Exception:
        return 0.0
    v = f0[(f0 > fmin) & (f0 < fmax)]
    return float(np.median(v)) if len(v) else 0.0


def hpss_pyin(x, sr, fmin=70, fmax=400):
    try:
        yh = librosa.effects.harmonic(x.astype(np.float32))
        f0, _, _ = librosa.pyin(yh, fmin=fmin, fmax=fmax, sr=sr,
                                frame_length=1024, hop_length=256)
    except Exception:
        return 0.0
    v = f0[(f0 > fmin) & (f0 < fmax)]
    return float(np.median(v)) if len(v) else 0.0


def ac_vote(x, sr, fmin=70, fmax=400):
    """多帧长自相关取众数（对谐波误锁更稳）"""
    cands = []
    for fl, hl in ((1024, 128), (2048, 256), (512, 64), (4096, 512)):
        f = ac(x, sr, fmin, fmax, octave_fix=False)
        if f > 0:
            cands.append(f)
    if not cands:
        return 0.0
    # 以 6% 为容差聚类，取最大簇的中位数
    cands = np.array(cands)
    best = None
    for c in cands:
        grp = cands[np.abs(cands - c) / c < 0.06]
        if best is None or len(grp) > len(best):
            best = grp
    return float(np.median(best))


METHODS = {
    "ac": ac,
    "ac_oct": lambda x, sr: ac(x, sr, octave_fix=True),
    "yin": yin,
    "pyin": pyin,
    "hpss_pyin": hpss_pyin,
    "ac_vote": ac_vote,
}


def synth_with_harmonics(f0_true, dur=1.0, sr=SR, n_harm=6, seed=0):
    """合成一个基频为 f0_true、含 n_harm 次谐波的信号，
    谐波相位随机 —— 用来检验算法在谐波干扰下的稳健性。"""
    rng = np.random.RandomState(seed)
    t = np.arange(int(sr * dur)) / sr
    x = np.zeros_like(t)
    for h in range(1, n_harm + 1):
        if f0_true * h < sr / 2:
            x += (1.0 / h) * np.sin(2 * np.pi * f0_true * h * t
                                    + rng.uniform(0, 2 * np.pi))
    return (0.5 * x / np.max(np.abs(x))).astype(np.float32)


def main():
    print("=" * 72)
    print("一、合成谐波信号：已知真值，各算法的测量误差")
    print("=" * 72)
    truths = [80.0, 100.0, 120.0, 150.0, 200.0, 250.0, 300.0, 350.0]
    errs = {k: [] for k in METHODS}
    for ft in truths:
        for seed in range(3):
            x = synth_with_harmonics(ft, seed=seed)
            for name, fn in METHODS.items():
                got = fn(x, SR)
                if got > 0:
                    errs[name].append(abs(got - ft) / ft * 100)
    print(f"{'算法':<12}{'中位误差%':>10}{'均值%':>9}{'p90%':>8}{'最大%':>8}")
    for name in METHODS:
        e = np.array(errs[name])
        print(f"{name:<12}{np.median(e):10.2f}{e.mean():9.2f}"
              f"{np.percentile(e,90):8.2f}{e.max():8.2f}")

    print("\n" + "=" * 72)
    print("二、真实语音：连续变调时读数的稳定性（越接近单调越好）")
    print("=" * 72)
    bm = json.load(open(BASE_F0, encoding="utf-8"))
    test_files = []
    for cid in ("F02", "F04", "M01", "M07", "M25"):
        p = f"{EMOTIONS}/{cid}/{cid}_nu.wav"
        if os.path.exists(p):
            test_files.append(p)

    def shift_raw(x, sr, factor):
        n_in = len(x)
        y = librosa.resample(x.astype(np.float32), orig_sr=sr,
                             target_sr=max(int(sr / factor), 1000))
        D = librosa.stft(y, n_fft=1024, hop_length=256)
        Ds = librosa.phase_vocoder(D, rate=1.0 / factor, hop_length=256,
                                   n_fft=1024)
        out = np.asarray(librosa.istft(Ds, hop_length=256, n_fft=1024),
                         dtype=np.float32)
        if len(out) < n_in:
            out = np.pad(out, (0, n_in - len(out)))
        elif len(out) > n_in:
            out = out[:n_in]
        return out

    factors = (0.90, 0.84, 0.80, 0.75, 0.70)
    print(f"{'文件':<8}" + "".join(f"{n[:9]:>11}" for n in METHODS))
    mono_scores = {k: [] for k in METHODS}
    for p in test_files:
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        cid = os.path.basename(p).split("_")[0]
        row = f"{cid:<8}"
        series = {}
        for name, fn in METHODS.items():
            cur = x.copy()
            vs = []
            for fac in factors:
                cur = shift_raw(cur, sr, fac)
                vs.append(fn(cur, sr))
            series[name] = vs
            # 单调性得分：应单调下降，相邻不升的次数越少越好
            asc = sum(1 for a, b in zip(vs, vs[1:]) if b > a * 1.02)
            mono_scores[name].append(asc)
            row += f"{vs[-1]:11.1f}"
        print(row)
    print("\n相邻步不降的次数（越少越稳，越小越好）：")
    for name in METHODS:
        print(f"  {name:<12}{sum(mono_scores[name])}")

    print("\n三、真实音频原文件上的读数（应与基线接近）")
    print(f"{'cid':<8}{'base':>8}" + "".join(f"{n[:9]:>11}" for n in METHODS))
    for p in test_files[:5]:
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        cid = os.path.basename(p).split("_")[0]
        base = bm.get(cid, 0.0)
        row = f"{cid:<8}{base:8.1f}"
        for name, fn in METHODS.items():
            row += f"{fn(x, sr):11.1f}"
        print(row)


if __name__ == "__main__":
    main()
