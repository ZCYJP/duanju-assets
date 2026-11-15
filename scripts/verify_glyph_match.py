#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""程序化汉字验收：模板匹配。

对每张带招牌的成图，用同一字体渲染预期文案作为模板，在牌面区域做
归一化互相关（极性无关），逐字给出匹配度。

  min_score >= 0.45  -> PASS（字在且字形对应）
  否则                -> FAIL（字缺失 / 被别的东西盖住 / 字形不对）

这是不依赖 OCR/VL 的客观判据。
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = '/usr/share/fonts/truetype/custom/SourceHanSansSC-Bold.otf'


def char_template(ch, size):
    f = ImageFont.truetype(FONT, size)
    pad = size
    img = Image.new('L', (size + pad * 2, size + pad * 2), 0)
    d = ImageDraw.Draw(img)
    d.text((pad, pad), ch, font=f, fill=255)
    return np.asarray(img)


def match_char(target, ch, sizes=range(36, 150, 4)):
    best = 0.0
    for size in sizes:
        tpl = char_template(ch, size)
        if tpl.shape[0] > target.shape[0] or tpl.shape[1] > target.shape[1]:
            continue
        r = cv2.matchTemplate(target.astype(np.float32), tpl.astype(np.float32),
                              cv2.TM_CCOEFF_NORMED)
        best = max(best, float(np.abs(r).max()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--scenes', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--threshold', type=float, default=0.45)
    args = ap.parse_args()

    with open(args.scenes, encoding='utf-8') as f:
        scenes = json.load(f)['scenes']
    items = []
    for sc in scenes:
        sign = sc.get('sign')
        if not sign:
            continue
        sid = sc['sid']
        path = os.path.join(args.dir, f'{sid}.png')
        if not os.path.exists(path):
            items.append({'sid': sid, 'expected': sign['text'], 'verdict': 'MISSING'})
            continue
        with Image.open(path) as im:
            img = im.convert('RGB')
        W, H = img.size
        x0, y0, x1, y1 = sign['box']
        l, t, r, b = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
        region = np.asarray(img.crop((l, t, r, b)).convert('L'), dtype=np.float32)
        scores = {ch: match_char(region, ch) for ch in sign['text']}
        mn = min(scores.values()) if scores else 0.0
        verdict = 'PASS' if mn >= args.threshold else 'FAIL'
        items.append({'sid': sid, 'expected': sign['text'], 'style': sign.get('style'),
                      'char_scores': {k: round(v, 3) for k, v in scores.items()},
                      'min_score': round(mn, 3), 'verdict': verdict})
        print(f"{verdict:5} {sid:26} {sign['text']!r:10} min={mn:.3f} "
              f"{[f'{k}:{v:.2f}' for k, v in scores.items()]}", flush=True)

    n_pass = sum(1 for r in items if r['verdict'] == 'PASS')
    summary = {'total': len(items), 'pass': n_pass, 'fail': len(items) - n_pass,
               'failed_sids': [r['sid'] for r in items if r['verdict'] != 'PASS']}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'items': items}, f, ensure_ascii=False, indent=1)
    print('=== summary ===', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
