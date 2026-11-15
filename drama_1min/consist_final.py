#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""成片跨镜场景一致性 —— 直接从成片抽帧测量"""
import os, sys, json, subprocess
import numpy as np
from PIL import Image

BASE = "/ssd2/duanju/drama_1min"
FINAL = f"{BASE}/final/drama_1min_h3.mp4"
TMP = f"{BASE}/qc_consist"
FFMPEG = "/ssd2/miniconda3/envs/videogen/bin/ffmpeg"

# 各镜在成片中的中点秒（片头1.0 + 常规镜4.9167，末镜5.1667）
starts = [1.0]
for i in range(11):
    starts.append(starts[-1] + 4.916667)
starts.append(starts[-1] + 4.916667)   # s12
names = ["s01_hook_fold","s02_ally_lawyer","s03_villain_seat","s04_exec_silent",
         "s05_break_alone","s06_garage_arrival","s07_rain_street","s08_villain_rage",
         "s09_lawyer_cards","s10_exec_reveal","s11_topfloor_reversal","s12_finale_hook"]

os.makedirs(TMP, exist_ok=True)
for f in os.listdir(TMP):
    os.remove(os.path.join(TMP, f))

print("=== 抽取各镜中点帧 ===")
paths = []
for n, st in zip(names, starts):
    mid = st + 2.0
    p = f"{TMP}/{n}.png"
    subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", f"{mid:.3f}", "-i", FINAL,
                    "-frames:v", "1", p], check=True)
    paths.append(p)
    print(f"  {n:<20} t={mid:6.2f}s")

def bg_hist(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)/255.0
    h, w, _ = a.shape
    mask = np.zeros((h, w), dtype=bool)
    mask[:int(h*0.33), :] = True
    mask[int(h*0.80):, :] = True
    mask[:, :int(w*0.20)] = True
    mask[:, int(w*0.80):] = True
    px = a[mask]
    if px.size == 0:
        px = a.reshape(-1,3)
    q = (px*16).astype(np.int32).clip(0,15)
    flat = q[:,0]*256 + q[:,1]*16 + q[:,2]
    hist = np.bincount(flat, minlength=4096).astype(np.float32)
    hist /= max(hist.sum(), 1)
    return hist, float(a.mean())

def cosine(a,b):
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))

hists, means = [], []
for p in paths:
    h, m = bg_hist(p)
    hists.append(h); means.append(m)

sim = np.zeros((12,12))
for i in range(12):
    for j in range(12):
        sim[i,j] = cosine(hists[i], hists[j])
iu = np.triu_indices(12, 1)
vals = sim[iu]
print(f"\n=== 成片跨镜背景一致性 ===")
print(f"两两相似度: 平均 {vals.mean():.3f}  最低 {vals.min():.3f}  最高 {vals.max():.3f}")
print(f"亮度极差: {max(means)-min(means):.3f}  (范围 {min(means):.3f}~{max(means):.3f})")
print("\n相似度矩阵:")
print("      " + " ".join(f"{i+1:>5d}" for i in range(12)))
for i in range(12):
    print(f"  {i+1:>2d} " + " ".join(f"{sim[i,j]:5.3f}" for j in range(12)))

# 同场景组（按剧本：boardroom/office/street/topfloor/garage）
groups = {
    "boardroom": [0,1,2,3,7,8],
    "topfloor":  [10,11],
    "street":    [6],
    "garage":    [5],
    "office":    [4],
}
print("\n同场景组内相似度:")
for g, idx in groups.items():
    if len(idx) < 2:
        continue
    sub = [sim[i,j] for a,i in enumerate(idx) for j in idx[a+1:]]
    print(f"  {g:<12} 平均 {np.mean(sub):.3f}  最低 {np.min(sub):.3f}  (n={len(idx)})")

json.dump({"sim_matrix": sim.round(4).tolist(), "means": means,
           "avg": float(vals.mean()), "min": float(vals.min())},
          open(f"{BASE}/final/consistency.json","w"), indent=2)
