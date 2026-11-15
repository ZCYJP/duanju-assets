#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
底图尺寸归一化 —— 统一 cover-crop 到 480x832 竖版（H3 FL2VA 的原生分辨率）。

底图是 832x1248（2:3），H3 工作分辨率 480x832（也是 2:3），比例一致，
所以这里只做等比 resize，不裁切，不损失构图。

背景：直接把非目标比例的图送进 FL2VA 会被拉伸压扁，导致人脸变形。
"""
import os
import glob
from PIL import Image

BASE = "/ssd2/duanju/drama_midnight47"
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
        if src_ratio > target_ratio:
            nw = int(sh * target_ratio)
            left = (sw - nw) // 2
            im = im.crop((left, 0, left + nw, sh))
        else:
            nh = int(sw / target_ratio)
            top = int((sh - nh) * 0.30)
            im = im.crop((0, top, sw, top + nh))
    im = im.resize((W, H), Image.LANCZOS)
    im.save(out)
    print(f"{name}: {sw}x{sh} -> {W}x{H}")
print(f"\n归一化完成 -> {DST}")
