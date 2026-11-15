#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判定牌面图层里到底有没有画出字：统计目标字色像素及其分布。"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, '/ssd2/duanju/scripts')
import paste_text  # noqa: E402

OUT = '/ssd2/duanju/_render_debug'


def analyze(path, fg, label):
    with Image.open(path) as im:
        arr = np.asarray(im.convert('RGBA'), dtype=np.int16)
    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    # 与目标字色的距离
    d = np.abs(rgb - np.array(fg, dtype=np.int16)).sum(axis=2)
    hit = (d < 60) & (alpha > 128)
    ys, xs = np.nonzero(hit)
    # 前景（任何与底色明显不同的像素）
    print(f'{label}: size={arr.shape[1]}x{arr.shape[0]} '
          f'字色像素={int(hit.sum())} '
          + (f'bbox=({xs.min()},{ys.min()})-({xs.max()},{ys.max()})' if hit.sum() else 'bbox=无'))
    return hit.sum()


def main():
    with open('/ssd2/duanju/scripts/scenes_b100.json', encoding='utf-8') as f:
        scenes = {s['sid']: s for s in json.load(f)['scenes']}
    # 1) 隔离渲染的 5 个样式
    for style in ('plaque', 'metal', 'paper', 'lightbox', 'stone'):
        fg = paste_text.STYLES[style]['fg']
        analyze(os.path.join(OUT, style, 'plate.png'), fg, f'[{style}] plate')
    # 2) 真实场景里的牌面（signed 图裁剪）
    print('--- 真实场景牌面 ---')
    for sid in ('bg100_academy', 'bg100_golden_hall', 'bg100_inn_hall', 'bg100_phone_booth'):
        sc = scenes.get(sid)
        if not sc or not sc.get('sign'):
            continue
        sign = sc['sign']
        with Image.open(f'/ssd2/duanju/scenes_b100_signed/{sid}.png') as im:
            img = im.convert('RGB')
        W, H = img.size
        x0, y0, x1, y1 = sign['box']
        crop = img.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
        crop.save(f'/ssd2/duanju/_render_debug/scene_{sid}.png')
        analyze(f'/ssd2/duanju/_render_debug/scene_{sid}.png', paste_text.STYLES[sign['style']]['fg'],
                f'[{sid}] {sign["style"]} exp={sign["text"]}')


if __name__ == '__main__':
    main()
