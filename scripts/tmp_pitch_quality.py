#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回正音质体检：确认相位声码器变调没有引入明显伪影。

背景
----
后期音高回正是必要手段（Qwen3-TTS 无法在生成端压住高唤醒情绪的音高漂移），
但相位声码器变调可能带来伪影：金属味、颤音（warble）、高频毛刺。
必须量化验证，否则「音高对了但听起来是假的」。

检查项（对比原文件与回正后文件）
--------------------------------
1. 波形完整：无 NaN/Inf，峰值归一化后不削波（clip ratio）
2. 频谱连续性：高频毛刺用「频谱一阶差分的抖动」衡量（spectral flux 噪声度）
3. 相位一致性：短时相位残差（warble 的典型特征是相邻帧相位跳变异常）
4. 能量包络：回正不应改变响度分布（rms 比值接近 1）
5. 时长：完全不变
6. f0 轨迹平滑度：变调后不应出现锯齿状抖动
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


def analyse(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    S = np.abs(librosa.stft(x, n_fft=1024, hop_length=256))
    # 高频毛刺：高频段 (>6kHz) 频谱帧间差分的变异系数
    hi = S[int(6000 * 1024 / sr):, :]
    hi_diff = np.diff(hi, axis=1)
    hf_noise = float(np.std(hi_diff) / (np.mean(hi_diff) + 1e-9))
    # 相位残差（warble 检测）：相邻帧相位差减去期望的线性相位推进
    D = librosa.stft(x, n_fft=1024, hop_length=256)
    ph = np.angle(D)
    dph = np.diff(ph, axis=1)
    # 期望相位推进（按帧移对应的频率）
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)[:, None]
    expected = 2 * np.pi * freqs * 256 / sr
    resid = np.abs(np.angle(np.exp(1j * (dph - expected))))
    phase_resid = float(np.mean(resid))
    # 能量包络
    rms = float(np.sqrt(np.mean(x ** 2)))
    # p95 幅度：比 rms 更接近感知响度（不受首尾静音干扰）
    p95 = float(np.percentile(np.abs(x), 95))
    # 削波比例
    clip = float(np.mean(np.abs(x) > 0.995))
    # 频谱质心（变调不应大幅改变）
    cent = float(librosa.feature.spectral_centroid(S=S, sr=sr).mean())
    # f0 轨迹平滑度：用 yin 粗估，取一阶差分的绝对值均值
    f0 = librosa.yin(x, fmin=60, fmax=400, sr=sr, frame_length=1024,
                     hop_length=256)
    v = f0[(f0 > 60) & (f0 < 400)]
    jitter = float(np.mean(np.abs(np.diff(v)))) if len(v) > 2 else 0.0
    return {
        "dur": len(x) / sr,
        "rms": rms,
        "p95": p95,
        "clip": clip,
        "hf_noise": hf_noise,
        "phase_resid": phase_resid,
        "centroid": cent,
        "jitter": jitter,
        "nan": bool(np.any(~np.isfinite(x))),
    }


def main():
    src_root = sys.argv[1] if len(sys.argv) > 1 else EMOTIONS
    fix_root = sys.argv[2] if len(sys.argv) > 2 else \
        "/ssd2/duanju/voices/emotions_fixed"
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 40

    base_map = json.load(open(BASE_F0, encoding="utf-8"))
    cands = []
    for p in sorted(glob.glob(f"{fix_root}/*/*.wav")):
        rel = os.path.relpath(p, fix_root)
        s = os.path.join(src_root, rel)
        if os.path.exists(s):
            cands.append((s, p))
    print(f"[scan] 找到 {len(cands)} 对可对比文件")
    if not cands:
        return
    step = max(1, len(cands) // n)
    sample = cands[::step][:n]

    rows = []
    for s, f in sample:
        try:
            a = analyse(s)
            b = analyse(f)
            cid = os.path.basename(s).split("_")[0]
            base = base_map.get(cid, 0.0)
            rows.append({"src": s, "fix": f, "cid": cid, "base": base,
                         "a": a, "b": b})
        except Exception as ex:
            print(f"  [err] {s}: {type(ex).__name__}: {ex}")

    print(f"\n{'指标':<14}{'原文件均值':>12}{'回正后均值':>12}{'变化':>10}")
    for k, label in (("rms", "响度rms"), ("p95", "响度p95"), ("clip", "削波率"),
                     ("hf_noise", "高频毛刺"), ("phase_resid", "相位残差"),
                     ("centroid", "频谱质心"), ("jitter", "f0抖动"),
                     ("dur", "时长")):
        va = np.mean([r["a"][k] for r in rows])
        vb = np.mean([r["b"][k] for r in rows])
        print(f"{label:<14}{va:12.4f}{vb:12.4f}{(vb/va if va else 0):10.3f}x")

    nans = sum(1 for r in rows if r["b"]["nan"])
    maxclip = max(r["b"]["clip"] for r in rows)
    durdiff = max(abs(r["b"]["dur"] - r["a"]["dur"]) for r in rows)
    print(f"\n[体检] NaN 文件数: {nans}")
    print(f"[体检] 回正后最大削波率: {maxclip:.5f}")
    print(f"[体检] 最大时长偏差: {durdiff:.6f}s")
    print(f"[体检] 样本 {len(rows)} 条")


if __name__ == "__main__":
    main()
