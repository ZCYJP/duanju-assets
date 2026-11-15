#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""极端漂移案例的变调方案探针。

背景
----
迭代闭环对中等漂移（r 1.5~2.5）效果良好，但极端案例（r>3.0）失败：
    M61 nu  r 3.45 -> 3.90（反而升高）
    M02 nu  r 3.13 -> 3.76
原因是大幅降调后：
  1) 自相关 f0 估计在低频区失真，闭环判断失准
  2) 多轮 phase vocoder 累积失真

本脚本针对极端案例，测试三种替代方案：
  A. 迭代闭环（现有）
  B. 一次性大幅降调（不做闭环，直接按理论倍率变调）
  C. 降采样域处理：先把音频重采样到更低采样率（天然降调），
     再用 phase vocoder 对齐时长

同时用一个【纯合成信号】做基准：对 220Hz 正弦，
先升到 3.45 倍（660Hz），再降回来，看各方案能否精确回到 220Hz。
这排除了语音本身谐波结构的干扰，能干净地测出算法能力。
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


def f0_ac(x, sr, octave_fix=True):
    frame = int(sr * 0.04)
    hop = int(sr * 0.02)
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
        seg_ac = ac[lo:hi]
        if len(seg_ac) == 0:
            continue
        k = int(np.argmax(seg_ac))
        f = sr / (lo + k)
        if octave_fix:
            peak = seg_ac[k]
            hp = len(seg_ac) // 2
            hpk = seg_ac[hp] if hp < len(seg_ac) else 0.0
            if hpk > 0.85 * peak and f > 82.5:
                f /= 2.0
        vals.append(f)
    return float(np.median(vals)) if vals else 0.0


def fit_len(out, n):
    if len(out) < n:
        return np.pad(out, (0, n - len(out)))
    return out[:n]


def shift_once(x, sr, factor, compensate=True):
    """单轮变调（保持时长）"""
    n = len(x)
    rms_in = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    y = librosa.resample(x.astype(np.float32), orig_sr=sr,
                         target_sr=max(int(sr / factor), 1000))
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    Ds = librosa.phase_vocoder(D, rate=1.0 / factor, hop_length=256, n_fft=1024)
    out = np.asarray(librosa.istft(Ds, hop_length=256, n_fft=1024), dtype=np.float32)
    out = fit_len(out, n)
    if compensate:
        ro = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
        if ro > 1e-8 and rms_in > 1e-8:
            out = out * (rms_in / ro)
        pk = float(np.max(np.abs(out)))
        if pk > 0.99:
            out = out * (0.99 / pk)
    return out


def plan_a_iterative(x, sr, base, target, max_rounds=10):
    """A. 迭代闭环"""
    y = x.astype(np.float32)
    f0 = f0_ac(y, sr)
    r = f0 / base if base else 0
    for _ in range(max_rounds):
        if abs(r - target) <= 0.03:
            break
        need = target / r if r > 0 else 1.0
        step = min(max(need, 0.75), 1.30)
        if abs(step - 1.0) < 0.01:
            break
        y = shift_once(y, sr, step)
        f0 = f0_ac(y, sr)
        r = f0 / base if base else 0
    return y, r


def plan_b_one_shot(x, sr, base, target, factor_cap=0.35):
    """B. 一次性大幅降调（按理论倍率，不闭环）"""
    f0 = f0_ac(x, sr)
    r = f0 / base if base else 0
    if r <= 0:
        return x, r
    need = target / r
    need = max(need, factor_cap)
    y = shift_once(x, sr, need)
    return y, f0_ac(y, sr) / base if base else 0


def plan_c_downsample(x, sr, base, target):
    """C. 降采样域处理：
    先重采样到 sr' = sr * (target/r)（天然降调且时长变化），
    再用 phase vocoder 把时长调回原长。
    关键：resample 到更低采样率后，原本超过 Nyquist 的高频被自然滤掉，
    这正是真实降调应有的效果（而 phase_vocoder 方案会产生混叠）。"""
    f0 = f0_ac(x, sr)
    r = f0 / base if base else 0
    if r <= 0:
        return x, r
    factor = target / r
    n = len(x)
    rms_in = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    # 目标采样率：直接降到 sr * factor（同时降调 + 降 Nyquist）
    sr2 = max(int(sr * factor), 4000)
    y = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=sr2)
    # 现在 y 长度 = n * sr2/sr = n*factor，播放时长不变但音高降了 factor
    # 用 phase vocoder 把时长压缩回原长（音高不变）
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    # y 长度是 n*factor，要变回 n 需要加速 factor 倍 -> rate = factor (>1 变快)
    Ds = librosa.phase_vocoder(D, rate=factor, hop_length=256, n_fft=1024)
    out = np.asarray(librosa.istft(Ds, hop_length=256, n_fft=1024), dtype=np.float32)
    out = fit_len(out, n)
    ro = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    if ro > 1e-8 and rms_in > 1e-8:
        out = out * (rms_in / ro)
    pk = float(np.max(np.abs(out)))
    if pk > 0.99:
        out = out * (0.99 / pk)
    return out, f0_ac(out, sr) / base if base else 0


print("=" * 70)
print("一、纯合成信号基准（排除语音谐波干扰）")
print("=" * 70)
SR = 24000
for up in (2.0, 3.45):
    t = np.arange(int(SR * 2.0)) / SR
    x = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    # 先升到 up 倍（模拟漂移）
    xup = shift_once(x, SR, up)
    f_up = f0_ac(xup, SR)
    print(f"\n升到 {up}x 后 f0={f_up:.1f}Hz（基准 220）")
    for name, fn in (("A迭代", plan_a_iterative), ("B一次性", plan_b_one_shot),
                     ("C降采样", plan_c_downsample)):
        y, r = fn(xup, SR, 220.0, 1.125)
        fy = f0_ac(y, SR)
        dur = len(y) / SR
        print(f"  {name:8s} f0={fy:7.1f}  目标220  误差={100*(fy-220)/220:+6.2f}%  "
              f"dur={dur:.3f}(原2.000)")

print("\n" + "=" * 70)
print("二、真实极端案例")
print("=" * 70)
base_map = json.load(open(BASE_F0, encoding="utf-8"))
targets = ["M61", "M02", "M47", "M12", "M65", "M25", "M60"]
for cid in targets:
    p = f"{EMOTIONS}/{cid}/{cid}_nu.wav"
    if not os.path.exists(p):
        continue
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    base = base_map.get(cid, 0.0)
    f0 = f0_ac(x, sr)
    r = f0 / base if base else 0
    print(f"\n{cid} base={base:.1f} 当前 f0={f0:.1f} r={r:.2f}")
    for name, fn in (("A迭代", plan_a_iterative), ("B一次性", plan_b_one_shot),
                     ("C降采样", plan_c_downsample)):
        y, rr = fn(x.copy(), sr, base, 1.125)
        fy = f0_ac(y, sr)
        dur = len(y) / sr
        print(f"  {name:8s} f0={fy:7.1f} r={rr:.3f} (目标1.125)  "
              f"dur={dur:.3f}(原{len(x)/sr:.3f})")
