#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨镜场景一致性检测 —— 同场景多镜是否像同一个空间。

指标：
  - 背景色直方图相似度（剔除人物中心区域，只取背景）
  - 全图色调/亮度均值方差
  - 两两相似度矩阵
"""
import sys, os, json
import numpy as np
from PIL import Image

def load(path):
    im = Image.open(path).convert("RGB")
    return np.asarray(im).astype(np.float32) / 255.0

def bg_hist(path):
    """取画面四边+四角区域（避开中心人物），做 HSV 风格色彩直方图"""
    a = load(path)
    h, w, _ = a.shape
    # 背景掩码：上1/3 + 下1/5 + 左右各1/5（避开中心人物区）
    mask = np.zeros((h, w), dtype=bool)
    mask[: int(h * 0.33), :] = True
    mask[int(h * 0.80):, :] = True
    mask[:, : int(w * 0.20)] = True
    mask[:, int(w * 0.80):] = True
    px = a[mask]
    if px.size == 0:
        px = a.reshape(-1, 3)
    # 16x16x16 颜色直方图
    q = (px * 16).astype(np.int32).clip(0, 15)
    flat = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    hist = np.bincount(flat, minlength=4096).astype(np.float32)
    hist /= max(hist.sum(), 1)
    return hist, float(a.mean()), float(a.std())

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

if __name__ == "__main__":
    # 参数：镜名:帧路径 ...（同组为同一场景）
    groups = json.loads(sys.argv[1])
    for gname, items in groups.items():
        print(f"\n=== {gname} ({len(items)} 镜) ===")
        hists = {}
        means = {}
        for name, path in items:
            if not os.path.exists(path):
                print(f"  {name}: 缺文件 {path}")
                continue
            hist, m, s = bg_hist(path)
            hists[name] = hist
            means[name] = (m, s)
        names = list(hists.keys())
        if len(names) < 2:
            continue
        print("  两两背景直方图余弦相似度：")
        print("        " + "".join(f"{n[:9]:>10}" for n in names))
        for a in names:
            row = "".join(f"{cosine(hists[a], hists[b]):>10.3f}" for b in names)
            print(f"  {a[:9]:<9}{row}")
        sims = [cosine(hists[a], hists[b]) for i, a in enumerate(names) for b in names[i+1:]]
        print(f"  平均相似度={np.mean(sims):.3f} 最低={np.min(sims):.3f} 最高={np.max(sims):.3f}")
        print("  亮度均值: " + " ".join(f"{n[:9]}={means[n][0]:.3f}" for n in names))
        lum = [means[n][0] for n in names]
        print(f"  亮度极差={max(lum)-min(lum):.3f}")
