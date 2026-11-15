#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：变调后 f0 读数与理论值的偏差方向。

现象
----
F02 nu：r 1.333 -> 1.383（rounds=1 就停）
按理论 factor=0.844 降调后 f0 应为 1.125，实际测出 1.383 —— 反而更高。

假设
----
pitch_shift_once 的降调路径是：
  resample(target_sr = sr/factor)  ->  音高降 factor、时长 ×1/factor
  phase_vocoder(rate = 1/factor)   ->  时长压回原长
若 phase_vocoder 的 rate 语义在【降采样后的新采样率】下与预期不同，
音高就不会按 factor 变化。

关键：resample 之后，音频的采样率变了（sr/factor），
但代码里 phase_vocoder 与 istft 仍按【原 sr】处理 ——
这在数学上是自洽的（频率轴按新 sr 解释），
但需要实测确认。

本脚本对真实音频逐级变调，打印 每一步的
  理论 f0、实测 f0、偏差率，
看偏差是系统性还是随机。
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
import librosa

sys.path.insert(0, "/ssd2/duanju/scripts")
from tmp_emotion_pitch_fix import f0_median  # noqa: E402

EMOTIONS = "/ssd2/duanju/voices/emotions"
BASE_F0 = "/ssd2/duanju/voices/base_f0.json"


def shift_raw(x, sr, factor):
    """与 pitch_shift_once 相同的路径，但不做 rms 补偿，便于单独看音高。"""
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


def main():
    bm = json.load(open(BASE_F0, encoding="utf-8"))
    for cid in ("F02", "F04", "M01", "M07"):
        p = f"{EMOTIONS}/{cid}/{cid}_nu.wav"
        if not os.path.exists(p):
            continue
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        base = bm.get(cid, 0.0)
        f0_0 = f0_median(x, sr, octave_fix=True)
        print(f"\n===== {cid} base={base:.1f} f0_0={f0_0:.1f} "
              f"r={f0_0/base:.3f} =====")
        cur = x.copy()
        for factor in (0.90, 0.84, 0.80, 0.75, 0.70):
            cur = shift_raw(cur, sr, factor)
            f = f0_median(cur, sr, octave_fix=True)
            theory = f0_0 * factor
            err = (f - theory) / theory * 100 if theory else 0
            print(f"  factor={factor:.2f}  实测f0={f:7.1f}  "
                  f"理论f0={theory:7.1f}  偏差={err:+7.2f}%  "
                  f"r={f/base:.3f}")

    # 反向：从低倍升回高倍，看升调是否对称
    print("\n===== 升调方向对称性测试（用合成信号）=====")
    SR = 24000
    t = np.arange(SR * 2) / SR
    x = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    f0_0 = f0_median(x, SR, octave_fix=True)
    print(f"  基准 f0={f0_0:.1f}")
    down = shift_raw(x, SR, 0.6)
    f_down = f0_median(down, SR, octave_fix=True)
    print(f"  降调 0.6 后 f0={f_down:.1f} (理论 {f0_0*0.6:.1f})")
    for up in (1.10, 1.25, 1.50, 1.667):
        cur = shift_raw(down, SR, up)
        f = f0_median(cur, SR, octave_fix=True)
        theory = f_down * up
        err = (f - theory) / theory * 100 if theory else 0
        print(f"    回升 x{up:.3f} -> 实测 {f:7.1f}  理论 {theory:7.1f}  "
              f"偏差 {err:+7.2f}%")


if __name__ == "__main__":
    main()
