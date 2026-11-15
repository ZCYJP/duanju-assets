#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单位测试：实测各种变调方法的真实音高/时长行为。

背景
----
tmp_emotion_pitch_fix.py 用「resample 降调 + phase_vocoder 压回时长」
做 pitch shift，实测降调幅度不足：
    F01 nu  factor_total=0.833 -> 期望 1.56*0.833=1.30，实测只到 1.39
说明 resample 与 phase_vocoder 串联时，两者对音高的作用不是简单相乘 ——
phase_vocoder 的 rate 参数会连带影响音高。

本脚本用一条已知音高的合成信号（220Hz 正弦）实测：
  方法1  resample(sr/f) + phase_vocoder(rate=1/f)
  方法2  resample(sr/f) + phase_vocoder(rate=f)
  方法3  resample(sr/f) + librosa.effects.time_stretch(rate=...)
  方法4  librosa.effects.pitch_shift(n_steps=...)
对每种方法测量：输出 f0 / 输入 f0（音高倍率）、输出时长 / 输入时长。
目标是找出「音高精确 ×f 且时长不变」的那一种。
"""
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os

SR = 24000
DUR = 2.0
F0 = 220.0


def make_tone(sr=SR, dur=DUR, f0=F0):
    t = np.arange(int(sr * dur)) / sr
    x = 0.5 * np.sin(2 * np.pi * f0 * t)
    return x.astype(np.float32)


def measure_f0(x, sr):
    hop = int(sr * 0.02)
    frame = int(sr * 0.04)
    lo, hi = int(sr / 400), int(sr / 70)
    vals = []
    for st in range(0, len(x) - frame, hop):
        seg = x[st:st + frame].astype(np.float64)
        if np.sqrt(np.mean(seg ** 2)) < 0.012:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[frame - 1:]
        if hi >= len(ac):
            continue
        vals.append(sr / (lo + int(np.argmax(ac[lo:hi]))))
    return float(np.median(vals)) if vals else 0.0


def fit_len(out, n):
    if len(out) < n:
        return np.pad(out, (0, n - len(out)))
    return out[:n]


def m1_resample_pv_rate_inv(x, sr, f):
    """当前实现：resample(sr/f) + phase_vocoder(rate=1/f)"""
    n = len(x)
    y = librosa.resample(x.astype(np.float32), orig_sr=sr,
                         target_sr=max(int(sr / f), 1000))
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    Ds = librosa.phase_vocoder(D, rate=1.0 / f, hop_length=256, n_fft=1024)
    out = np.asarray(librosa.istft(Ds, hop_length=256, n_fft=1024), dtype=np.float32)
    return fit_len(out, n), sr


def m2_resample_pv_rate_fwd(x, sr, f):
    """resample(sr/f) + phase_vocoder(rate=f)"""
    n = len(x)
    y = librosa.resample(x.astype(np.float32), orig_sr=sr,
                         target_sr=max(int(sr / f), 1000))
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    Ds = librosa.phase_vocoder(D, rate=f, hop_length=256, n_fft=1024)
    out = np.asarray(librosa.istft(Ds, hop_length=256, n_fft=1024), dtype=np.float32)
    return fit_len(out, n), sr


def m3_resample_pv_nofit(x, sr, f):
    """resample(sr/f) + phase_vocoder(rate=1/f)，不裁剪长度"""
    y = librosa.resample(x.astype(np.float32), orig_sr=sr,
                         target_sr=max(int(sr / f), 1000))
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    Ds = librosa.phase_vocoder(D, rate=1.0 / f, hop_length=256, n_fft=1024)
    out = np.asarray(librosa.istft(Ds, hop_length=256, n_fft=1024), dtype=np.float32)
    return out, sr


def m4_pitch_shift(x, sr, f):
    """librosa.effects.pitch_shift（内部走 phase vocoder）"""
    n_steps = int(round(12 * np.log2(1.0 / f)))
    out = librosa.effects.pitch_shift(x.astype(np.float32), sr=sr,
                                      n_steps=n_steps)
    return fit_len(out, len(x)), sr


def m5_resample_only(x, sr, f):
    """纯 resample（只降调不管时长）"""
    y = librosa.resample(x.astype(np.float32), orig_sr=sr,
                         target_sr=max(int(sr / f), 1000))
    return y, max(int(sr / f), 1000)


def m6_resample_then_time_stretch(x, sr, f):
    """resample(sr/f) + librosa.effects.time_stretch"""
    n = len(x)
    y = librosa.resample(x.astype(np.float32), orig_sr=sr,
                         target_sr=max(int(sr / f), 1000))
    try:
        out = librosa.effects.time_stretch(y, rate=1.0 / f)
    except Exception as ex:
        return None, f"ERR {type(ex).__name__}: {ex}"
    return fit_len(np.asarray(out, dtype=np.float32), n), sr


METHODS = {
    "M1 resample+pv(1/f)": m1_resample_pv_rate_inv,
    "M2 resample+pv(f)": m2_resample_pv_rate_fwd,
    "M3 resample+pv(1/f) 不裁剪": m3_resample_pv_nofit,
    "M4 pitch_shift": m4_pitch_shift,
    "M5 纯resample": m5_resample_only,
    "M6 resample+time_stretch": m6_resample_then_time_stretch,
}

x0 = make_tone()
f0_in = measure_f0(x0, SR)
dur_in = len(x0) / SR
print(f"输入: f0={f0_in:.1f}Hz dur={dur_in:.3f}s")
print(f"{'方法':32s} {'音高倍率':>10s} {'时长倍率':>10s} {'实测f0':>10s}")
for f in (0.80, 0.60):
    print(f"\n--- 目标 factor={f} ---")
    for name, fn in METHODS.items():
        try:
            out, sr2 = fn(x0.copy(), SR, f)
            if out is None:
                print(f"{name:32s}  {sr2}")
                continue
            fo = measure_f0(out, sr2)
            do = len(out) / sr2
            print(f"{name:32s} {fo/f0_in:10.3f} {do/dur_in:10.3f} {fo:10.1f}")
        except Exception as ex:
            print(f"{name:32s}  ERR {type(ex).__name__}: {ex}")
