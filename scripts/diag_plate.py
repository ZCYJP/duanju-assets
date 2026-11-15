#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判定牌面里到底有没有文字墨迹，并定位墨迹的空间分布。

思路：牌面底色是相对均匀的（木纹/石纹有噪声但中值稳定）。
统计每行/每列上"偏离中值"的像素数，文字笔画应形成若干个
连续的高墨迹列（横排时每字一组竖条），竖排时形成若干行。

同时输出 ASCII 字符画用于人眼确认。
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def ink_profile(arr, axis):
    """按行/列统计偏离中值的像素占比。arr: 灰度 float 2D"""
    med = np.median(arr)
    dev = np.abs(arr - med) > 38
    if axis == 'col':
        return dev.mean(axis=0)
    return dev.mean(axis=1)


def ascii_art(img, w=96, h=40):
    g = np.asarray(img.convert('L'), dtype=np.float32)
    H, W = g.shape
    # 分块取均值
    yi = np.linspace(0, H, h + 1).astype(int)
    xi = np.linspace(0, W, w + 1).astype(int)
    out = []
    ramp = ' .:-=+*#%@'
    for y in range(h):
        row = []
        for x in range(w):
            blk = g[yi[y]:max(yi[y] + 1, yi[y + 1]), xi[x]:max(xi[x] + 1, xi[x + 1])]
            v = blk.mean() if blk.size else 255
            row.append(ramp[min(9, int((255 - v) / 256 * 10))])
        out.append(''.join(row))
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plate_dir', required=True)
    ap.add_argument('--scenes', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', default='')
    ap.add_argument('--art', default='', help='给这个 sid 打 ASCII 字符画')
    args = ap.parse_args()

    with open(args.scenes, encoding='utf-8') as f:
        scenes = json.load(f)['scenes']
    only = set(x for x in args.only.split(',') if x)

    items = []
    for sc in scenes:
        sign = sc.get('sign')
        if not sign:
            continue
        sid = sc['sid']
        if only and sid not in only:
            continue
        path = os.path.join(args.plate_dir, f'{sid}.png')
        if not os.path.exists(path):
            continue
        with Image.open(path) as im:
            plate = im.convert('RGB')
        g = np.asarray(plate.convert('L'), dtype=np.float32)
        H, W = g.shape

        cols = ink_profile(g, 'col')
        rows = ink_profile(g, 'row')
        # 墨迹最强的列段（横排文字：4 个字 = 4 段）
        col_thr = max(0.05, float(cols.max()) * 0.45)
        row_thr = max(0.05, float(rows.max()) * 0.45)
        col_hit = int((cols > col_thr).sum())
        row_hit = int((rows > row_thr).sum())

        # 中央区域的墨迹覆盖率（排除边框）
        m = int(min(H, W) * 0.18)
        core = g[m:H - m, m:W - m]
        core_med = np.median(core)
        core_ink = float((np.abs(core - core_med) > 38).mean())

        items.append({'sid': sid, 'expected': sign['text'], 'style': sign.get('style'),
                      'vertical': bool(sign.get('vertical')),
                      'plate_size': [W, H], 'col_ink_px': col_hit, 'row_ink_px': row_hit,
                      'core_ink_ratio': round(core_ink, 4),
                      'max_col': round(float(cols.max()), 3),
                      'max_row': round(float(rows.max()), 3)})
        print(f"{sid:26} {sign['text']!r:12} {sign.get('style'):9} "
              f"size={W}x{H} colhit={col_hit:3} rowhit={row_hit:3} "
              f"core={core_ink:.3f} maxcol={cols.max():.2f} maxrow={rows.max():.2f}", flush=True)

        if args.art and sid == args.art:
            print('\n===== ASCII ART', sid, sign['text'], '=====')
            print(ascii_art(plate))
            print('===== col profile =====')
            print(''.join('|' if v > col_thr else ('.' if v > col_thr / 2 else ' ')
                          for v in cols))
            print('===== row profile =====')
            print(''.join('|' if v > row_thr else ('.' if v > row_thr / 2 else ' ')
                          for v in rows))

    summary = {'total': len(items),
               'core_ink_gt_0.05': sum(1 for r in items if r['core_ink_ratio'] > 0.05),
               'core_ink_gt_0.15': sum(1 for r in items if r['core_ink_ratio'] > 0.15)}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'items': items}, f, ensure_ascii=False, indent=1)
    print('=== summary ===', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
