#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调试：检查 signed 图的牌面区域是否真的写入了文字。

输出：
  1) 牌面区域在 raw / signed 两图之间的像素差异统计
  2) 裁剪出的牌面图（供 VL 直接读）
"""
import json
import os
import sys

import numpy as np
from PIL import Image

SCENES = '/ssd2/duanju/scripts/scenes_b100.json'
RAW = '/ssd2/duanju/scenes_b100'
SIGNED = '/ssd2/duanju/scenes_b100_signed'
OUT = '/ssd2/duanju/_sign_debug'


def main():
    ids = sys.argv[1].split(',') if len(sys.argv) > 1 else ['bg100_academy', 'bg100_golden_hall',
                                                            'bg100_executive', 'bg100_inn_hall']
    with open(SCENES, encoding='utf-8') as f:
        scenes = {s['sid']: s for s in json.load(f)['scenes']}
    os.makedirs(OUT, exist_ok=True)
    for sid in ids:
        sc = scenes.get(sid)
        if not sc or not sc.get('sign'):
            print(f'{sid}: 无 sign 定义')
            continue
        sign = sc['sign']
        with Image.open(f'{RAW}/{sid}.png') as im:
            raw = im.convert('RGB')
        with Image.open(f'{SIGNED}/{sid}.png') as im:
            signed = im.convert('RGB')
        W, H = raw.size
        x0, y0, x1, y1 = sign['box']
        l, t, r, b = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
        a = np.asarray(raw.crop((l, t, r, b)), dtype=np.int16)
        c = np.asarray(signed.crop((l, t, r, b)), dtype=np.int16)
        diff = np.abs(a - c).sum(axis=2)
        changed = (diff > 18).mean() * 100
        print(f'{sid}: exp={sign["text"]!r} box={sign["box"]} size={(r-l, b-t)} '
              f'变化像素占比={changed:.1f}% 平均差={diff.mean():.1f} 最大差={diff.max()}')
        crop = signed.crop((max(0, l - 10), max(0, t - 10), min(W, r + 10), min(H, b + 10)))
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        crop.save(f'{OUT}/{sid}_sign.png')
        print(f'   裁剪牌面 -> {OUT}/{sid}_sign.png  ({crop.size})')


if __name__ == '__main__':
    main()
