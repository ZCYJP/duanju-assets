#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证「升调校准测量法」：先升调到高位再测，反推真实倍率。

背景
----
v4 实测发现：pYIN 在【降调后】的音频上读数会系统性偏低
（F17 降调 0.60 倍后读 0.881，真实应为 1.125，偏低 22%）。
而 pYIN 在【升调后】的音频上读数精确
（往返验证：x1.4 误差 2.2%、x2.0 误差 2.0%、x2.6 误差 0.8%）。

于是得到一个「测量只走升调方向」的办法：
    z = 变调(x, K=1.6)        # 升调
    f0_hi = pYIN(z)            # 高位测量（精确区）
    r_true = f0_hi / K / base  # 反推真实倍率

本脚本在已知真实倍率的合成场景下检验该法的准确度，
并与「直接降调后测量」对比。
"""
import glob
import json
import os

import numpy as np
import soundfile as sf
import librosa

VOICES = "/ssd2/duanju/voices"
FMIN, FMAX = 70.0, 800.0


def f0_pyin(x, sr):
    try:
        f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=FMIN, fmax=FMAX,
                                sr=sr, frame_length=1024, hop_length=256)
    except Exception:
        return 0.0
    v = f0[(f0 > FMIN) & (f0 < FMAX)]
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


def r_direct(x, sr, base):
    """直接测量法：pYIN(x) / base"""
    return f0_pyin(x, sr) / base


def r_calib(x, sr, base, K=1.6):
    """升调校准法：pYIN(shift(x,K)) / K / base"""
    z = shift(x, sr, K)
    return f0_pyin(z, sr) / K / base


def main():
    files = sorted(glob.glob(f"{VOICES}/samples/*_voice.wav") +
                   glob.glob(f"{VOICES}/samples_ext/*_voice.wav"))
    bm = json.load(open(f"{VOICES}/base_f0_pyin.json", encoding="utf-8"))
    # 取 10 条覆盖不同基线
    sel = files[::15][:10]

    factors = (0.5, 0.6, 0.75, 1.0, 1.4, 2.0)

    print("=" * 84)
    print("一、已知真实倍率场景下的测量误差")
    print("   注意：真实倍率是【先变调产生】的，base 用原始音频的 pyin 读数")
    print("=" * 84)
    print(f"{'真实倍率':<10}{'直接测量中位':>14}{'直接测量p90':>14}"
          f"{'校准测量中位':>14}{'校准测量p90':>14}")
    for fac in factors:
        ed, ec = [], []
        for p in sel:
            cid = os.path.basename(p).split("_")[0]
            base = bm.get(cid, 0)
            if base <= 0:
                continue
            x, sr = sf.read(p, dtype="float32")
            if x.ndim > 1:
                x = x[:, 0]
            y = shift(x, sr, fac)
            rd = r_direct(y, sr, base)
            rc = r_calib(y, sr, base)
            if rd > 0:
                ed.append(abs(rd - fac) / fac * 100)
            if rc > 0:
                ec.append(abs(rc - fac) / fac * 100)
        ed, ec = np.array(ed), np.array(ec)
        print(f"x{fac:<9.2f}{np.median(ed):13.1f}%{np.percentile(ed,90):13.1f}%"
              f"{np.median(ec):13.1f}%{np.percentile(ec,90):13.1f}%")

    print("\n" + "=" * 84)
    print("二、K 值扫描（用真实倍率 0.6 和 1.4 两个场景，选误差最小的 K）")
    print("=" * 84)
    for K in (1.3, 1.5, 1.6, 1.8, 2.0):
        errs = []
        for fac in (0.5, 0.6, 0.75, 1.4):
            for p in sel:
                cid = os.path.basename(p).split("_")[0]
                base = bm.get(cid, 0)
                if base <= 0:
                    continue
                x, sr = sf.read(p, dtype="float32")
                if x.ndim > 1:
                    x = x[:, 0]
                y = shift(x, sr, fac)
                rc = r_calib(y, sr, base, K=K)
                if rc > 0:
                    errs.append(abs(rc - fac) / fac * 100)
        e = np.array(errs)
        print(f"K={K:<5.1f} 中位 {np.median(e):6.2f}%  p90 {np.percentile(e,90):6.2f}%"
              f"  最大 {e.max():6.2f}%")

    print("\n" + "=" * 84)
    print("三、对真实情绪音频（F17 案例）用两种方法读数")
    print("=" * 84)
    for cid, emo in (("F17", "nu"), ("F02", "nu"), ("M07", "nu"), ("M65", "xi")):
        p = f"{VOICES}/emotions/{cid}/{cid}_{emo}.wav"
        if not os.path.exists(p):
            continue
        base = bm.get(cid, 0)
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        print(f"{cid}_{emo}: base={base:.1f}  直接测量 r={r_direct(x,sr,base):.3f}"
              f"  校准测量 r={r_calib(x,sr,base):.3f}")


if __name__ == "__main__":
    main()
