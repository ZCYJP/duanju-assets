#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：合成牌面在成图中的实际像素落点 vs 期望 box。

方法：成图 - 原图 = 差分图。牌面是唯一变化源，所以差分像素的包围盒
就是牌面真实落点。把它和 spec['box']（归一化）换算后的像素盒对比，
即可判定：
  A) 落点基本吻合 -> 合成位置正确，问题在"字太小/被背景干扰"
  B) 落点严重偏移 -> box 与实际牌位不符，验收裁剪读错区域
  C) 差分像素极少 -> 牌面根本没画进去（alpha 被丢弃等）

用法:
  diag_overlay.py --scenes scenes_b100.json --raw <原图目录> --signed <贴字后目录>
                   --out diag_overlay.json
"""
import argparse
import json
import os

import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--signed', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', default='')
    args = ap.parse_args()

    with open(args.scenes, encoding='utf-8') as f:
        scenes = json.load(f)
    if isinstance(scenes, dict):
        scenes = scenes.get('scenes', [])
    only = set(x for x in args.only.split(',') if x)

    items = []
    for sc in scenes:
        sid = sc['sid']
        sign = sc.get('sign')
        if only and sid not in only:
            continue
        if not sign or not sign.get('text'):
            continue
        raw_p = os.path.join(args.raw, f'{sid}.png')
        sig_p = os.path.join(args.signed, f'{sid}.png')
        if not (os.path.exists(raw_p) and os.path.exists(sig_p)):
            items.append({'sid': sid, 'status': 'MISSING'})
            continue

        with Image.open(raw_p) as im:
            raw = np.asarray(im.convert('RGB'), dtype=np.int16)
        with Image.open(sig_p) as im:
            sig = np.asarray(im.convert('RGB'), dtype=np.int16)
        if raw.shape != sig.shape:
            items.append({'sid': sid, 'status': 'SIZE_MISMATCH',
                          'raw': list(raw.shape), 'signed': list(sig.shape)})
            continue

        H, W = raw.shape[:2]
        diff = np.abs(raw - sig).sum(axis=2)
        mask = diff > 24  # 每通道和 > 24 才算变化（滤掉 JPEG/PNG 噪声）

        rec = {'sid': sid, 'expected': sign['text'], 'style': sign.get('style'),
               'vertical': bool(sign.get('vertical')), 'box': sign['box'],
               'img': [W, H]}

        if mask.sum() == 0:
            rec['status'] = 'NO_CHANGE'
            items.append(rec)
            print(f"NO_CHANGE {sid}", flush=True)
            continue

        ys, xs = np.nonzero(mask)
        ax0, ax1 = int(xs.min()), int(xs.max())
        ay0, ay1 = int(ys.min()), int(ys.max())

        x0, y0, x1, y1 = sign['box']
        bx0, by0 = int(x0 * W), int(y0 * H)
        bx1, by1 = int(x1 * W), int(y1 * H)

        # 交集 / 并集（IoU）
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        area_a = (ax1 - ax0) * (ay1 - ay0)
        area_b = (bx1 - bx0) * (by1 - by0)
        union = area_a + area_b - inter
        iou = inter / union if union else 0.0

        # 牌面真实中心是否落在期望 box 内
        acx, acy = (ax0 + ax1) / 2, (ay0 + ay1) / 2
        in_box = bx0 <= acx <= bx1 and by0 <= acy <= by1

        # 变化像素中有多少落在期望 box 内
        in_box_px = int(mask[by0:by1, bx0:bx1].sum())
        frac = in_box_px / int(mask.sum())

        rec.update({'status': 'OK', 'actual_bbox': [ax0, ay0, ax1, ay1],
                    'expect_bbox_px': [bx0, by0, bx1, by1],
                    'actual_size': [ax1 - ax0, ay1 - ay0],
                    'expect_size': [bx1 - bx0, by1 - by0],
                    'iou': round(iou, 3), 'center_in_box': bool(in_box),
                    'frac_change_in_box': round(frac, 3),
                    'change_px': int(mask.sum()),
                    'actual_center': [int(acx), int(acy)]})
        items.append(rec)
        print(f"{sid:28} iou={iou:.3f} in_box={in_box} frac={frac:.3f} "
              f"actual={rec['actual_size']} expect={rec['expect_size']} "
              f"ac={rec['actual_center']} bc=[{(bx0+bx1)//2},{(by0+by1)//2}]", flush=True)

    n_ok = sum(1 for r in items if r.get('status') == 'OK')
    n_center_out = sum(1 for r in items if r.get('status') == 'OK' and not r['center_in_box'])
    n_low_iou = sum(1 for r in items if r.get('status') == 'OK' and r['iou'] < 0.5)
    n_nochange = sum(1 for r in items if r.get('status') == 'NO_CHANGE')
    summary = {'total': len(items), 'ok': n_ok, 'center_outside_box': n_center_out,
               'iou_below_0.5': n_low_iou, 'no_change': n_nochange}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'items': items}, f, ensure_ascii=False, indent=1)
    print('=== summary ===', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
