#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""f0 估计器实验室：在【真实语音】上比较候选估计器的可靠性。

背景
----
问题：pYIN / 自相关在部分真实语音上会锁到半频（octave-down），
导致回正闭环把音高修到错误位置。

决定性检验方式
--------------
往返检验（round-trip）：
    1. 取真实语音
    2. 用已验证精确的变调算子按【已知倍率】变调
    3. 用各估计器测量，算「测得倍率 / 真实倍率」
    真实倍率 = 1 说明该估计器可靠。
这把「估计器误差」与「变调算子误差」彻底解耦。

候选估计器
----------
1. ac         自相关取最大峰（经典，最易八度错）
2. pyin       librosa.pYIN
3. yin_thr    自研 YIN + 绝对阈值 + 「首个低于阈值的谷」
   —— YIN 的绝对阈值规则本就是为了修八度错：
      取 d'(tau) 首次低于阈值的【局部极小】，
      而不是全局最小 -> 天然偏向真正的基周期（更早的 tau）
4. yin_thr_v  yin_thr 的候选 + 谐波投票复核（用长时谱）
"""
import glob
import json
import os

import numpy as np
import soundfile as sf
import librosa

VOICES = "/ssd2/duanju/voices"
FMIN, FMAX = 65.0, 700.0


# ------------------------------------------------------------ 变调算子
def shift(x, sr, factor):
    """已验证精确的变调算子（tmp_pitch_unit.py：误差 <0.3%）。"""
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


# ------------------------------------------------------------ 估计器
def est_ac(x, sr, fmin=FMIN, fmax=FMAX):
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
        vals.append(sr / (lo + int(np.argmax(s))))
    return float(np.median(vals)) if vals else 0.0


def est_pyin(x, sr, fmin=FMIN, fmax=FMAX):
    try:
        f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=fmin, fmax=fmax,
                                sr=sr, frame_length=1024, hop_length=256)
    except Exception:
        return 0.0
    v = f0[(f0 > fmin) & (f0 < fmax)]
    return float(np.median(v)) if len(v) else 0.0


def _yin_diff(seg, tau_max):
    """YIN 差分函数 d(tau)，用自相关加速。

    d(tau) = sum_j (x[j] - x[j+tau])^2
           = P0 + Ptau - 2*ACF(tau)
    其中 P0 = sum x[j]^2, Ptau = sum x[j+tau]^2（滑动窗和），
    ACF(tau) = sum_j x[j]*x[j+tau]。
    """
    n = len(seg)
    seg = seg - seg.mean()
    p0 = float(np.dot(seg, seg))
    # 自相关（未归一化）
    acf = np.correlate(seg, seg, "full")[n - 1:]
    if len(acf) <= tau_max:
        return None
    # Ptau: sum_{j=0}^{n-1-tau} x[j+tau]^2 = cumsum of squares 后向差分
    sq = np.concatenate(([0.0], np.cumsum(seg ** 2)))
    ptau = sq[n:] - sq[:n]          # 长度 n，ptau[tau] = sum x[tau:]^2
    d = p0 + ptau[:tau_max + 1] - 2.0 * acf[:tau_max + 1]
    return np.maximum(d, 0.0)


def _parabolic(y, i):
    """抛物线插值细化极小点。"""
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    denom = a - 2 * b + c
    if abs(denom) < 1e-12:
        return float(i)
    delta = 0.5 * (a - c) / denom
    if abs(delta) > 1.0:
        return float(i)
    return float(i) + delta


def est_yin_thr(x, sr, fmin=FMIN, fmax=FMAX, threshold=0.15):
    """自研 YIN：绝对阈值 + 首个谷。

    相对 librosa.yin 的差别：
      - librosa.yin 默认用「全局最小」策略（center-cancel=False 时其实
        也是阈值法，但阈值/搜索窗细节不同，且对 65~700Hz 宽窗不利）
      - 本实现显式把 d' 归一化，并严格取「首个低于阈值的局部极小」，
        这是 YIN 论文的原始设计，目的就是避免八度错误
        （半频会让 d' 出现次级极小，阈值法会选中更早的真基周期）。
    """
    frame = 1024
    hop = 256
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
        # 累积均值归一化差分 d'
        cum = np.cumsum(d[1:])
        taus = np.arange(1, len(d))
        dprime = np.ones_like(d)
        dprime[1:] = d[1:] * taus / np.maximum(cum, 1e-12)
        # 绝对阈值：找首个 d' < threshold 的位置
        below = np.where(dprime[tau_min:tau_max + 1] < threshold)[0]
        if len(below) == 0:
            # 无低于阈值 -> 取 d' 最小点
            seg_d = dprime[tau_min:tau_max + 1]
            i = int(np.argmin(seg_d)) + tau_min
        else:
            i = int(below[0]) + tau_min
            # 从该点向后找局部极小
            while i + 1 <= tau_max and dprime[i + 1] < dprime[i]:
                i += 1
        # 抛物线细化
        tau = _parabolic(dprime, i)
        if tau <= 0:
            continue
        f = sr / tau
        if fmin <= f <= fmax:
            vals.append(f)
    return float(np.median(vals)) if vals else 0.0


ESTIMATORS = {
    "ac": est_ac,
    "pyin": est_pyin,
    "yin_thr": est_yin_thr,
}


# ------------------------------------------------------------ 主流程
def main():
    bm = json.load(open(f"{VOICES}/base_f0_pyin.json", encoding="utf-8"))
    files = sorted(glob.glob(f"{VOICES}/samples/*_voice.wav") +
                   glob.glob(f"{VOICES}/samples_ext/*_voice.wav"))
    by_g = {"M": [], "F": []}
    for p in files:
        cid = os.path.basename(p).split("_")[0]
        if cid[0] in by_g and len(by_g[cid[0]]) < 8:
            by_g[cid[0]].append(p)
    test_files = by_g["M"] + by_g["F"]
    print(f"样本 {len(test_files)} 条（男 {len(by_g['M'])} / 女 {len(by_g['F'])}）")

    factors = (0.5, 0.6, 0.75, 1.0, 1.4, 2.0)

    print("\n" + "=" * 76)
    print("一、往返误差：|测得倍率 - 真实倍率| / 真实倍率")
    print("=" * 76)
    print(f"{'估计器':<12}" + "".join(f"{('x'+str(f)):>12}" for f in factors)
          + f"{'中位':>10}{'p90':>10}{'最大':>10}")
    for name, fn in ESTIMATORS.items():
        errs = []
        cells = ""
        for fac in factors:
            a = []
            for p in test_files:
                cid = os.path.basename(p).split("_")[0]
                base = bm.get(cid, 0)
                if base <= 0:
                    continue
                x, sr = sf.read(p, dtype="float32")
                if x.ndim > 1:
                    x = x[:, 0]
                y = shift(x, sr, fac)
                f0m = fn(y, sr)
                f0b = fn(x, sr)
                if f0m <= 0 or f0b <= 0:
                    continue
                a.append(abs(f0m / f0b - fac) / fac * 100)
            errs += a
            cells += f"{(np.median(a) if a else -1):11.1f}%"
        e = np.array(errs)
        print(f"{name:<12}{cells}{np.median(e):9.1f}%{np.percentile(e,90):9.1f}%"
              f"{e.max():9.1f}%")

    print("\n" + "=" * 76)
    print("二、真实情绪音频：三种估计器给出的原始倍率（应接近，且无半频）")
    print("=" * 76)
    print(f"{'cid':<8}{'base':>8}" + "".join(f"{n:>12}" for n in ESTIMATORS))
    for cid in ("F17", "F02", "M07", "M65", "F01", "M03", "M12"):
        p = f"{VOICES}/emotions/{cid}/{cid}_nu.wav"
        if not os.path.exists(p):
            continue
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        base = bm.get(cid, 0)
        row = f"{cid:<8}{base:8.1f}"
        for name, fn in ESTIMATORS.items():
            row += f"{fn(x, sr)/base if base else 0:12.3f}"
        print(row)

    print("\n" + "=" * 76)
    print("三、F17 案例：单次大幅降调后各估计器的读数")
    print("   （真相：F17_nu 真实 f0 约 310Hz，降调 0.698 后应约 217Hz）")
    print("=" * 76)
    p = f"{VOICES}/emotions/F17/F17_nu.wav"
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    base = bm["F17"]
    for fac in (1.0, 0.85, 0.698, 0.6, 0.5):
        y = shift(x, sr, fac)
        row = f"  因子{fac:<6.3f} 期望 f0={310.7*fac:6.1f}Hz  "
        for name, fn in ESTIMATORS.items():
            row += f"{name}={fn(y, sr):7.1f} "
        print(row)


if __name__ == "__main__":
    main()
