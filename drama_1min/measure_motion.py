#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运动量测量 —— 对比 H3 与 Wan 的动态幅度。"""
import sys, os, subprocess, json
import numpy as np

FFMPEG = "/ssd2/miniconda3/envs/videogen/bin/ffmpeg"


def load_frames(path, max_frames=140, w=240):
    tmp = "/tmp/_mv_frames"
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    cmd = [FFMPEG, "-y", "-v", "error", "-i", path,
           "-vf", f"fps=24,scale={w}:-2", "-frames:v", str(max_frames),
           os.path.join(tmp, "f_%04d.png")]
    subprocess.run(cmd, check=True, capture_output=True)
    arrs = []
    for f in sorted(os.listdir(tmp)):
        out = subprocess.run(
            [FFMPEG, "-v", "error", "-i", os.path.join(tmp, f), "-f", "rawvideo",
             "-pix_fmt", "gray", "-"], capture_output=True)
        arrs.append(np.frombuffer(out.stdout, dtype=np.uint8).astype(np.float32) / 255.0)
    return arrs


def metrics(arrs):
    if len(arrs) < 2:
        return {}
    diffs = [float(np.mean(np.abs(arrs[i + 1] - arrs[i]))) for i in range(len(arrs) - 1)]
    frozen = sum(1 for d in diffs if d < 0.002)
    return {
        "frames": len(arrs),
        "avg_diff": round(float(np.mean(diffs)), 5),
        "max_diff": round(float(np.max(diffs)), 5),
        "frozen": frozen,
        "moving_ratio": round(1 - frozen / len(diffs), 4),
    }


if __name__ == "__main__":
    results = {}
    for path in sys.argv[1:]:
        name = os.path.basename(path).replace(".mp4", "")
        arrs = load_frames(path)
        m = metrics(arrs)
        results[name] = m
        print(f"{name:<34} 帧数={m['frames']:>4} 均差={m['avg_diff']:.5f} "
              f"最大差={m['max_diff']:.5f} 冻结={m['frozen']:>3} 运动占比={m['moving_ratio']:.3f}",
              flush=True)
    print()
    print(json.dumps(results, ensure_ascii=False, indent=2))
