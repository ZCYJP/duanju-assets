#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""变调算子的可组合性验证：连续两次变调是否等价于一次相乘变调。

关键问题
--------
v5 的「升调校准」测量法在 F17 上给出荒谬结果：
    原文件 f0 ≈ 307Hz（原生 pYIN 读数）
    升调 1.6 倍 -> 期望 492Hz
    实际校准测量给出 266.8*1.6 = 427Hz（偏低 13%）
    再降调 0.698 倍后（期望 214.5Hz）
    校准测量给出 121.7Hz（偏低 43%！）

若「升调后再测量」这条路径本身有系统性缺陷，
则所有基于校准的测量都不可信，必须换方案。

本脚本做三组实测：
  A. 原生测量 vs 升调后测量（在多倍率上）
  B. 降调后原生测量（验证算子降调方向是否精确）
  C. 复合变调：shift(shift(x, K), f) 的实际效果
"""
import json
import os

import numpy as np
import soundfile as sf
import librosa

VOICES = "/ssd2/duanju/voices"


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


def pyin(x, sr):
    try:
        f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=65, fmax=700, sr=sr,
                                frame_length=1024, hop_length=256)
    except Exception:
        return 0.0
    v = f0[(f0 > 65) & (f0 < 700)]
    return float(np.median(v)) if len(v) else 0.0


def main():
    for cid in ("F17", "F02", "M07"):
        p = f"{VOICES}/emotions/{cid}/{cid}_nu.wav"
        if not os.path.exists(p):
            continue
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        f0_0 = pyin(x, sr)
        print("=" * 76)
        print(f"{cid}_nu  sr={sr}  时长={len(x)/sr:.2f}s  原生 f0={f0_0:.1f}Hz")
        print("=" * 76)

        print("\nA. 升调后原生测量（验证升调算子 + 测量是否一致）")
        print(f"{'K':>7}{'期望f0':>11}{'实测f0':>11}{'比值':>10}{'误差%':>10}")
        for K in (1.2, 1.4, 1.6, 1.8, 2.0):
            y = shift(x, sr, K)
            f0m = pyin(y, sr)
            exp = f0_0 * K
            print(f"{K:7.2f}{exp:11.1f}{f0m:11.1f}"
                  f"{(f0m/f0_0 if f0_0 else 0):10.3f}"
                  f"{((f0m/exp-1)*100 if exp else 0):10.1f}")

        print("\nB. 降调后原生测量（验证降调算子）")
        print(f"{'f':>7}{'期望f0':>11}{'实测f0':>11}{'比值':>10}{'误差%':>10}")
        for f in (0.85, 0.7, 0.6, 0.5):
            y = shift(x, sr, f)
            f0m = pyin(y, sr)
            exp = f0_0 * f
            print(f"{f:7.2f}{exp:11.1f}{f0m:11.1f}"
                  f"{(f0m/f0_0 if f0_0 else 0):10.3f}"
                  f"{((f0m/exp-1)*100 if exp else 0):10.1f}")

        print("\nC. 复合变调：shift(shift(x,K), f) 后再原生测量")
        print("   期望 = f0_0 * K * f")
        print(f"{'K':>6}{'f':>7}{'期望f0':>11}{'实测f0':>11}{'误差%':>10}")
        for K, f in ((1.6, 1.0), (1.6, 0.9), (1.6, 0.7), (2.0, 0.7),
                     (1.3, 0.8)):
            z = shift(shift(x, sr, K), sr, f)
            f0m = pyin(z, sr)
            exp = f0_0 * K * f
            print(f"{K:6.2f}{f:7.2f}{exp:11.1f}{f0m:11.1f}"
                  f"{((f0m/exp-1)*100 if exp else 0):10.1f}")

        print("\nD. 降调后【再升调回原高】再测量（校准法的理想情形）")
        print(f"{'f':>7}{'期望f0':>11}{'回升后实测':>13}{'误差%':>10}")
        for f in (0.85, 0.7, 0.6, 0.5):
            y = shift(x, sr, f)
            z = shift(y, sr, 1.0 / f)
            f0m = pyin(z, sr)
            exp = f0_0
            print(f"{f:7.2f}{exp:11.1f}{f0m:13.1f}"
                  f"{((f0m/exp-1)*100 if exp else 0):10.1f}")
        print()


if __name__ == "__main__":
    main()
