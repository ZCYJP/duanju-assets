#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""变调-测量 往返验证：在真实语音上直接检验「测量口径」的可靠性。

背景
----
之前只在【合成信号】上验证过各 f0 算法的绝对精度（tmp_f0_robust.py），
但真实语音有呼吸声、摩擦音、浊音不连续等复杂性，合成信号不能代表。

本脚本做的是决定性的往返检验：
    1. 取角色【基线主音色】；
    2. 用与回正完全相同的变调算子，按已知倍率变调；
    3. 用各候选口径测量，算出「测得倍率 / 真实倍率」；
    4. 真实倍率 = 1 说明该口径在真实语音上完全可靠。

因为基线与回正后都用同一把尺子（同一算法），若某口径在所有倍率上
都能正确读出真实倍率，则闭环回正必然收敛。

候选口径
--------
    ac400     自相关，搜索窗 70~400Hz（原方案）
    yin400    librosa.yin，搜索窗 70~400Hz
    yin800    librosa.yin，搜索窗 70~800Hz（可测到漂移区）
    pyin800   librosa.pyin，搜索窗 70~800Hz
    yin1200   librosa.yin，搜索窗 70~1200Hz（极端漂移区）
"""
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
import librosa

VOICES = "/ssd2/duanju/voices"
BASE_F0 = f"{VOICES}/base_f0.json"


def _frames(x, sr, fmin, fmax):
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


def ac400(x, sr):
    return _frames(x, sr, 70, 400)


def _librosa_f0(x, sr, fmin, fmax, method="yin"):
    try:
        if method == "yin":
            f0 = librosa.yin(x.astype(np.float32), fmin=fmin, fmax=fmax, sr=sr,
                             frame_length=1024, hop_length=256)
        else:
            f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=fmin, fmax=fmax,
                                    sr=sr, frame_length=1024, hop_length=256)
    except Exception:
        return 0.0
    v = f0[(f0 > fmin) & (f0 < fmax)]
    return float(np.median(v)) if len(v) else 0.0


def yin400(x, sr):
    return _librosa_f0(x, sr, 70, 400, "yin")


def yin800(x, sr):
    return _librosa_f0(x, sr, 70, 800, "yin")


def pyin800(x, sr):
    return _librosa_f0(x, sr, 70, 800, "pyin")


def yin1200(x, sr):
    return _librosa_f0(x, sr, 70, 1200, "yin")


METHODS = {
    "ac400": ac400,
    "yin400": yin400,
    "yin800": yin800,
    "pyin800": pyin800,
    "yin1200": yin1200,
}


def shift_raw(x, sr, factor):
    """与回正脚本完全一致的变调算子（含 rms 幅度补偿）。"""
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


def main():
    bm = json.load(open(BASE_F0, encoding="utf-8"))
    files = sorted(glob.glob(f"{VOICES}/samples/*_voice.wav") +
                   glob.glob(f"{VOICES}/samples_ext/*_voice.wav"))
    # 男女各取若干，覆盖不同基线音高
    by_gender = {"M": [], "F": []}
    for p in files:
        cid = os.path.basename(p).split("_")[0]
        if cid[0] in by_gender and len(by_gender[cid[0]]) < 6:
            by_gender[cid[0]].append(p)
    test_files = by_gender["M"] + by_gender["F"]
    print(f"测试样本 {len(test_files)} 条（男 {len(by_gender['M'])} / 女 {len(by_gender['F'])}）")

    factors = (0.5, 0.7, 1.4, 2.0, 2.6)

    # ---------- 1. 逐倍率逐口径的相对误差 ----------
    print("\n" + "=" * 78)
    print("一、往返相对误差：|测得倍率 - 真实倍率| / 真实倍率  （越低越好）")
    print("=" * 78)
    err = {m: {f: [] for f in factors} for m in METHODS}
    for p in test_files:
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        for fac in factors:
            y = shift_raw(x, sr, fac)
            for mname, fn in METHODS.items():
                f0_meas = fn(y, sr)
                if f0_meas <= 0:
                    continue
                # 基线用同口径测量同一份原文件（保证同尺子）
                r_meas = f0_meas / fn(x, sr)
                err[mname][fac].append(abs(r_meas - fac) / fac * 100)

    header = f"{'口径':<10}" + "".join(f"{('x'+str(f)):>12}" for f in factors) + f"{'总体中位':>12}{'总体p90':>10}"
    print(header)
    for mname in METHODS:
        allv = []
        cells = ""
        for f in factors:
            a = np.array(err[mname][f])
            allv += list(a)
            cells += f"{(np.median(a) if len(a) else -1):11.1f}%"
        allv = np.array(allv)
        print(f"{mname:<10}{cells}{np.median(allv):11.1f}%{np.percentile(allv,90):9.1f}%")

    # ---------- 2. 逐样本明细（看最大偏差） ----------
    print("\n" + "=" * 78)
    print("二、最大单样本偏差（%），定位最差情况")
    print("=" * 78)
    for mname in METHODS:
        worst = 0.0
        worst_info = ""
        for p in test_files:
            x, sr = sf.read(p, dtype="float32")
            if x.ndim > 1:
                x = x[:, 0]
            cid = os.path.basename(p).split("_")[0]
            for fac in factors:
                y = shift_raw(x, sr, fac)
                f0m = METHODS[mname](y, sr)
                f0b = METHODS[mname](x, sr)
                if f0m <= 0 or f0b <= 0:
                    continue
                d = abs(f0m / f0b - fac) / fac * 100
                if d > worst:
                    worst = d
                    worst_info = f"{cid} x{fac}: 测得{f0m/f0b:.2f}"
        print(f"{mname:<10} 最差 {worst:6.1f}%   {worst_info}")

    # ---------- 3. 基线重算对照 ----------
    print("\n" + "=" * 78)
    print("三、基线重算：各口径在 150 条主音色上的读数 vs 现有 base_f0.json")
    print("=" * 78)
    res = {}
    for mname in METHODS:
        vals = []
        for p in files:
            cid = os.path.basename(p).split("_")[0]
            x, sr = sf.read(p, dtype="float32")
            if x.ndim > 1:
                x = x[:, 0]
            v = METHODS[mname](x, sr)
            vals.append((cid, v))
        res[mname] = dict(vals)
        arr = np.array([v for _, v in vals])
        print(f"{mname:<10} 均值 {arr.mean():7.1f}  中位 {np.median(arr):7.1f}  "
              f"最小 {arr.min():6.1f}  最大 {arr.max():7.1f}")

    print("\n与 base_f0.json(自相关口径) 的偏差：")
    base_arr = np.array([bm[c] for c, _ in
                          [(os.path.basename(p).split('_')[0], p) for p in files]])
    for mname in METHODS:
        arr = np.array([res[mname][os.path.basename(p).split("_")[0]] for p in files])
        rel = (arr - base_arr) / base_arr * 100
        print(f"{mname:<10} 中位 {np.median(rel):+7.1f}%  "
              f"绝对值p90 {np.percentile(np.abs(rel),90):6.1f}%  "
              f"最大 {np.max(np.abs(rel)):6.1f}%")

    json.dump({m: res[m] for m in METHODS},
              open(f"{VOICES}/base_f0_by_method.json", "w"), indent=2)
    print(f"\n-> {VOICES}/base_f0_by_method.json")


if __name__ == "__main__":
    main()
