#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F17 失败案例深度诊断。

现象：F17_nu  base=165.5
      r_before(校准)=1.612  r_before(直接)=1.877
      开环 factor=0.6979（=1.125/1.612）→ 预期 r=1.125
      实测 r_after(校准)=0.735   ← 差了 35%，远超校准法 p90 的 5.7%

可能原因：
  A. F17 音频含多说话人/非语音段，f0 轨迹多峰
  B. F17 的 base 主音色与情绪音频的音域差异大
  C. 该角色基线样本本身有测量歧义

本脚本输出：
  1. F17 base 与 F17_nu 的逐帧 pyin f0 轨迹分布
  2. F17_nu 在连续降调序列下的 direct / calib 读数
  3. 用「同一文件内相对变化」检验算子是否真的生效
"""
import json
import os

import numpy as np
import soundfile as sf
import librosa

VOICES = "/ssd2/duanju/voices"
FMIN, FMAX = 70.0, 800.0
K = 1.6


def f0_pyin(x, sr):
    try:
        f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=FMIN, fmax=FMAX,
                                sr=sr, frame_length=1024, hop_length=256)
    except Exception:
        return None
    return f0


def med(a):
    v = a[(a > FMIN) & (a < FMAX)]
    return float(np.median(v)) if len(v) else 0.0


def shift(x, sr, factor):
    if abs(factor - 1.0) < 1e-6:
        return x.astype(np.float32)
    n_in = len(x)
    y = librosa.resample(x.astype(np.float32), orig_sr=sr,
                         target_sr=max(int(sr / factor), 1000))
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    Ds = librosa.phase_vocoder(D, rate=1.0 / factor, hop_length=256, n_fft=1024)
    out = np.asarray(librosa.istft(Ds, hop_length=256, n_fft=1024), dtype=np.float32)
    if len(out) < n_in:
        out = np.pad(out, (0, n_in - len(out)))
    elif len(out) > n_in:
        out = out[:n_in]
    return out


def hist(f0, label):
    if f0 is None:
        print(f"  {label}: 无数据")
        return
    v = f0[(f0 > FMIN) & (f0 < FMAX)]
    if not len(v):
        print(f"  {label}: 全部越界")
        return
    # 分簇统计众数
    hist, edges = np.histogram(v, bins=20)
    top = np.argsort(hist)[::-1][:3]
    clusters = [f"[{edges[i]:.0f}-{edges[i+1]:.0f}]x{hist[i]}" for i in top]
    print(f"  {label}: n={len(v)} 中位={np.median(v):.1f} "
          f"p10={np.percentile(v,10):.1f} p90={np.percentile(v,90):.1f} "
          f"众数簇 {clusters}")


def main():
    cid = "F17"
    bm = json.load(open(f"{VOICES}/base_f0_pyin.json", encoding="utf-8"))
    bc = json.load(open(f"{VOICES}/base_f0_calib.json", encoding="utf-8"))
    base = bm[cid]
    print(f"{cid}: base_direct={base}  base_calib={bc[cid]}")

    # --- 找 base 文件 ---
    bp = None
    for p in (f"{VOICES}/samples/{cid}_*_voice.wav",
              f"{VOICES}/samples_ext/{cid}_*_voice.wav"):
        g = sorted(__import__("glob").glob(p))
        if g:
            bp = g[0]
            break
    xb, srb = sf.read(bp, dtype="float32")
    if xb.ndim > 1:
        xb = xb[:, 0]
    print(f"\n一、base 主音色 {os.path.basename(bp)}  时长 {len(xb)/srb:.2f}s")
    fb = f0_pyin(xb, srb)
    hist(fb, "base f0 轨迹")

    ep = f"{VOICES}/emotions/{cid}/{cid}_nu.wav"
    xe, sre = sf.read(ep, dtype="float32")
    if xe.ndim > 1:
        xe = xe[:, 0]
    print(f"\n二、情绪音频 {cid}_nu  时长 {len(xe)/sre:.2f}s")
    fe = f0_pyin(xe, sre)
    hist(fe, "nu f0 轨迹")

    print(f"\n三、连续降调序列下的读数（base={base}）")
    print(f"{'累计因子':>10}{'direct读数':>12}{'direct/base':>13}"
          f"{'calib读数':>12}{'calib/base':>12}{'相对直读比':>13}")
    cur = xe.copy()
    for fac in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        if fac != 1.0:
            cur = shift(cur, sre, fac)
        fd = med(f0_pyin(cur, sre))
        fc = med(f0_pyin(shift(cur, sre, K), sre)) / K
        print(f"{fac:10.2f}{fd:12.1f}{fd/base:13.3f}{fc:12.1f}{fc/base:12.3f}"
              f"{(fc/fd if fd else 0):13.3f}")

    print("\n四、direct 与 calib 在「同一文件不同变调」下的相对一致性")
    print("   （若二者比值恒定，说明只是整体偏移；若跳变，说明存在谐波误锁）")
    cur = xe.copy()
    ratios = []
    for fac in (1.0, 0.9, 0.8, 0.7, 0.6):
        if fac != 1.0:
            cur = shift(cur, sre, fac)
        fd = med(f0_pyin(cur, sre))
        fc = med(f0_pyin(shift(cur, sre, K), sre)) / K
        ratios.append(fc / fd if fd else 0)
    print("   calib/direct 比值序列:", " ".join(f"{r:.3f}" for r in ratios))
    print("   极差 %.1f%%" % ((max(ratios) - min(ratios)) / np.mean(ratios) * 100))

    print("\n五、其它失败/成功角色对比（同口径检查）")
    for c2 in ("F02", "M07", "M65", "F01"):
        p2 = f"{VOICES}/emotions/{c2}/{c2}_nu.wav"
        if not os.path.exists(p2):
            continue
        x2, sr2 = sf.read(p2, dtype="float32")
        if x2.ndim > 1:
            x2 = x2[:, 0]
        f0t = f0_pyin(x2, sr2)
        b2 = bm[c2]
        fd = med(f0t)
        fc = med(f0_pyin(shift(x2, sr2, K), sr2)) / K
        print(f"  {c2}: base={b2:6.1f} direct={fd/b2:.3f} calib={fc/b2:.3f} "
              f"比值={fc/fd if fd else 0:.3f}")


if __name__ == "__main__":
    main()
