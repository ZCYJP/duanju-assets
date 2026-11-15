#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""裁剪牌面区域并高倍放大，用于 VL 精确回读汉字。

同时输出两种视图：
  <sid>.png   原始彩图放大 4 倍
  <sid>_bw.png 灰度对比度拉伸 + 放大 4 倍（字形轮廓更清晰）
"""
import argparse
import json
import os

import numpy as np
from PIL import Image, ImageOps

SCENES = '/ssd2/duanju/scripts/scenes_b100.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--scenes', default=SCENES)
    ap.add_argument('--out', required=True)
    ap.add_argument('--zoom', type=int, default=4)
    args = ap.parse_args()

    with open(args.scenes, encoding='utf-8') as f:
        scenes = json.load(f)['scenes']
    os.makedirs(args.out, exist_ok=True)
    n = 0
    for sc in scenes:
        sign = sc.get('sign')
        if not sign:
            continue
        sid = sc['sid']
        path = os.path.join(args.dir, f'{sid}.png')
        if not os.path.exists(path):
            continue
        with Image.open(path) as im:
            img = im.convert('RGB')
        W, H = img.size
        x0, y0, x1, y1 = sign['box']
        l, t = int(x0 * W), int(y0 * H)
        r, b = int(x1 * W), int(y1 * H)
        pad = int((r - l) * 0.06)
        crop = img.crop((max(0, l - pad), max(0, t - pad),
                         min(W, r + pad), min(H, b + pad)))
        big = crop.resize((crop.width * args.zoom, crop.height * args.zoom), Image.LANCZOS)
        big.save(os.path.join(args.out, f'{sid}.png'))

        g = ImageOps.grayscale(big)
        arr = np.asarray(g, dtype=np.float32)
        lo, hi = np.percentile(arr, 5), np.percentile(arr, 95)
        if hi - lo < 12:
            hi = lo + 12
        arr = np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(args.out, f'{sid}_bw.png'))
        n += 1
    print(f'裁剪放大 {n} 张 -> {args.out}')


if __name__ == '__main__':
    main()
