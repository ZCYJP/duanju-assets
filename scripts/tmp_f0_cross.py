#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双算法交叉验证 f0 测量口径：自相关 vs librosa.yin。

背景
----
音高回正的闭环依赖 f0 测量。若测量本身有偏，闭环就会把音频"修"到错误的位置：
  * 自相关 + 70~400Hz 搜索窗：大幅降调后基频低于下限，锁到 2/3 倍谐波，
    测出的 f0 反而变高 -> 闭环发散（实测 M65 2.84 -> 4.69）
  * librosa.yin：自带倍频折叠，低音区更稳，但在很低的基线上
    分辨率下降，可能偏低估

本脚本对【同一批真实情绪音频】用两种算法各测一遍，
并用第三条独立证据（pyin / 或直接对比回正前后的一致性）交叉判断，
输出两者的偏差分布，确定回正脚本该用哪一个。
"""
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
import librosa


def f0_ac(x, sr, fmin=70, fmax=400):
    """原自相关实现（含简单的倍频折叠检查）"""
    frame = int(sr * 0.04)
    hop = int(sr * 0.02)
    lo, hi = int(sr / fmax), int(sr / fmin)
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


def f0_ac_octave(x, sr, fmin=60, fmax=400):
    """自相关 + 倍频折叠修正：
    若最高相关峰低于半频处峰值的 85%，说明可能锁到高次谐波，折半。"""
    frame = int(sr * 0.04)
    hop = int(sr * 0.02)
    lo, hi = int(sr / fmax), int(sr / fmin)
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
        k = int(np.argmax(seg_ac))
        peak = seg_ac[k]
        # 检查半频处（对应 2 倍频）的峰
        half_pos = len(seg_ac) // 2
        half_peak = seg_ac[half_pos] if half_pos < len(seg_ac) else 0.0
        f = sr / (lo + k)
        if half_peak > 0.85 * peak and f > fmin * 1.5:
            f = f / 2.0
        vals.append(f)
    return float(np.median(vals)) if vals else 0.0


def f0_yin(x, sr):
    try:
        f0 = librosa.yin(x.astype(np.float32), fmin=45, fmax=450, sr=sr,
                         frame_length=1024, hop_length=256)
    except Exception:
        return 0.0
    v = f0[(f0 > 45) & (f0 < 450)]
    return float(np.median(v)) if len(v) else 0.0


def f0_pyin(x, sr):
    try:
        f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=45, fmax=450,
                                sr=sr, frame_length=1024, hop_length=256)
    except Exception:
        return 0.0
    v = f0[(f0 > 45) & (f0 < 450)]
    return float(np.median(v)) if len(v) else 0.0


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/ssd2/duanju/voices/emotions"
    emo = sys.argv[2] if len(sys.argv) > 2 else "nu"
    base_map = json.load(open("/ssd2/duanju/voices/base_f0.json", encoding="utf-8"))

    files = sorted(glob.glob(f"{root}/*/*_{emo}.wav"))
    print(f"[scan] {len(files)} 个 {emo} 文件\n")
    print(f"{'cid':<7}{'base':>7}{'ac':>8}{'ac_oct':>8}{'yin':>8}"
          f"{'pyin':>8}{'r_ac':>7}{'r_oct':>7}{'r_yin':>7}{'r_pyin':>7}")
    rows = []
    for p in files:
        cid = os.path.basename(p).split("_")[0]
        base = base_map.get(cid, 0.0)
        if base <= 0:
            continue
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        a = f0_ac(x, sr)
        ao = f0_ac_octave(x, sr)
        y = f0_yin(x, sr)
        py = f0_pyin(x, sr)
        rows.append({"cid": cid, "base": base, "ac": a, "ac_oct": ao,
                     "yin": y, "pyin": py})
        print(f"{cid:<7}{base:7.1f}{a:8.1f}{ao:8.1f}{y:8.1f}{py:8.1f}"
              f"{a/base:7.2f}{ao/base:7.2f}{y/base:7.2f}{py/base:7.2f}")

    print("\n=== 各算法相对基线倍率的统计 ===")
    for k in ("ac", "ac_oct", "yin", "pyin"):
        r = np.array([x[k] / x["base"] for x in rows])
        print(f"  {k:8s} mean={r.mean():6.2f} median={np.median(r):6.2f} "
              f"p95={np.percentile(r,95):6.2f} max={r.max():6.2f}")
    # 两两差异
    print("\n=== 相对偏差（%）===")
    for k in ("ac_oct", "yin", "pyin"):
        d = np.array([(x[k] - x["ac"]) / x["ac"] for x in rows if x["ac"] > 0])
        print(f"  {k:8s} vs ac: mean={d.mean()*100:6.1f}%  "
              f"median={np.median(d)*100:6.1f}%  "
              f"|diff|>20% 的比例={np.mean(np.abs(d)>0.2)*100:5.1f}%")


if __name__ == "__main__":
    main()
