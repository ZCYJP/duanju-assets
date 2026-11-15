#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 H3 合成音频：配音是否可闻、能量分布。"""
import subprocess, sys, os, json
import numpy as np

FF = "/ssd2/miniconda3/envs/videogen/bin/ffmpeg"
FP = "/ssd2/miniconda3/envs/videogen/bin/ffprobe"
SR = 16000


def to_wav(src, dst):
    subprocess.run([FF, "-y", "-v", "error", "-i", src, "-vn", "-ac", "1",
                    "-ar", str(SR), dst], check=True)


def load(p):
    out = subprocess.run([FF, "-v", "error", "-i", p, "-f", "f32le", "-ac", "1",
                          "-ar", str(SR), "-"], capture_output=True, check=True)
    return np.frombuffer(out.stdout, dtype=np.float32)


def rms_db(x):
    r = float(np.sqrt(np.mean(x ** 2)))
    return 20 * np.log10(max(r, 1e-9))


def analyze(path, label, voice_spans):
    wav = "/tmp/_diag.wav"
    to_wav(path, wav)
    x = load(wav)
    print(f"\n=== {label} ({os.path.basename(path)}) ===")
    print(f"  总长 {len(x)/SR:.2f}s  全局RMS {rms_db(x):.1f} dBFS  峰值 {20*np.log10(max(abs(x).max(),1e-9)):.1f} dBFS")
    for i, (a, b) in enumerate(voice_spans):
        seg = x[int(a * SR):int(b * SR)]
        if len(seg) == 0:
            continue
        pre = x[max(0, int((a - 0.5) * SR)):int(a * SR)]
        post = x[int(b * SR):int(min(len(x) / SR, b + 0.4)) * SR]
        print(f"  句{i+1} 配音区间 {a:.2f}~{b:.2f}s  RMS={rms_db(seg):>6.1f} dBFS"
              f" | 前静默 {rms_db(pre) if len(pre) else float('nan'):>6.1f}"
              f" | 后静默 {rms_db(post) if len(post) else float('nan'):>6.1f}")


if __name__ == "__main__":
    final = sys.argv[1]
    tlj = sys.argv[2]
    tl = json.load(open(tlj, encoding="utf-8"))
    spans = [(s["sub_start"], s["sub_end"]) for s in tl["shots"]]
    analyze(final, "成片", spans)
    # 逐镜片段
    for s in tl["shots"][:3]:
        p = f"/ssd2/duanju/drama_1min/compose_h3_work/seg_{s['shot']}.mp4"
        if os.path.exists(p):
            analyze(p, s["shot"], [(0.30, 0.30 + s["voice_dur"])])
