#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
底图尺寸归一化 —— 统一 cover-crop 到 480x832 竖版。

背景：s11/s12 的底图是 1392x752 横版，直接送入 FL2VA 会被拉伸压扁，
导致人脸变形（Vision 检测不到人脸）。本脚本把全部底图统一为 480x832，
横版用 center-crop 保留中心人物。
"""
import os, glob
from PIL import Image

BASE = "/ssd2/duanju/drama_1min"
SRC = f"{BASE}/frames"
DST = f"{BASE}/frames_norm"
W, H = 480, 832

os.makedirs(DST, exist_ok=True)
for p in sorted(glob.glob(f"{SRC}/*.png")):
    name = os.path.basename(p)
    out = os.path.join(DST, name)
    im = Image.open(p).convert("RGB")
    sw, sh = im.size
    target_ratio = W / H
    src_ratio = sw / sh
    if abs(src_ratio - target_ratio) > 1e-3:
        # cover-crop：按目标比例裁掉多余部分，保留中心
        if src_ratio > target_ratio:
            # 源更宽 -> 裁左右
            nw = int(sh * target_ratio)
            left = (sw - nw) // 2
            im = im.crop((left, 0, left + nw, sh))
        else:
            # 源更高 -> 裁上下（保上 1/3，脸在上方）
            nh = int(sw / target_ratio)
            top = int((sh - nh) * 0.30)
            im = im.crop((0, top, sw, top + nh))
    im = im.resize((W, H), Image.LANCZOS)
    im.save(out)
    print(f"{name}: {sw}x{sh} -> {W}x{H}")
print(f"\n归一化完成 -> {DST}")
