#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把牌面区域二值化成 ASCII 字符画，直接在终端里"看"字形。"""
import json
import sys

import numpy as np
from PIL import Image

SCENES = '/ssd2/duanju/scripts/scenes_b100.json'
RAMP = ' .:-=+*#%@'


def render(path, box, cols=110, invert=False):
    with Image.open(path) as im:
        img = im.convert('L')
    W, H = img.size
    x0, y0, x1, y1 = box
    l, t, r, b = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
    crop = img.crop((l, t, r, b))
    a = np.asarray(crop, dtype=np.float32)
    rows = max(1, int(cols * a.shape[0] / a.shape[1] / 2.1))
    yi = (np.linspace(0, a.shape[0] - 1, rows)).astype(int)
    xi = (np.linspace(0, a.shape[1] - 1, cols)).astype(int)
    small = a[np.ix_(yi, xi)]
    lo, hi = np.percentile(small, 3), np.percentile(small, 97)
    if hi - lo < 1e-3:
        hi = lo + 1.0
    norm = np.clip((small - lo) / (hi - lo), 0, 1)
    if invert:
        norm = 1 - norm
    idx = (norm * (len(RAMP) - 1)).astype(int)
    print(f'--- {path.split("/")[-1]}  box={box}  {crop.size} ---')
    for row in idx:
        print(''.join(RAMP[v] for v in row))


def main():
    ids = sys.argv[1].split(',')
    d = sys.argv[2] if len(sys.argv) > 2 else '/ssd2/duanju/scenes_b100_final'
    with open(SCENES, encoding='utf-8') as f:
        scenes = {s['sid']: s for s in json.load(f)['scenes']}
    for sid in ids:
        sc = scenes.get(sid)
        if not sc or not sc.get('sign'):
            print(sid, '无 sign')
            continue
        render(f'{d}/{sid}.png', sc['sign']['box'])


if __name__ == '__main__':
    main()
