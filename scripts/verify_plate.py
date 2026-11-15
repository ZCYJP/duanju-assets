#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对「独立牌面图层」做 OCR 回读（渲染层验收）。

牌面图层是从 paste_text.py 导出的纯 PNG（无场景背景），
读得出 = 字体渲染层完全正确；读不出 = 渲染层本身有问题。

用法:
  verify_plate.py --plate_dir /ssd2/duanju/_diag_plate \
                  --scenes /ssd2/duanju/scripts/scenes_b100.json \
                  --out /ssd2/duanju/bg100_verify_plate.json
"""
import argparse
import difflib
import json
import os
import re

os.environ.setdefault('OMP_THREAD_LIMIT', '1')

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def norm(s):
    if not s:
        return ''
    s = re.sub(r'\s+', '', s.strip())
    s = ''.join(ch for ch in s if ch.isprintable())
    return s


def sim(a, b):
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plate_dir', required=True)
    ap.add_argument('--scenes', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no_gpu', action='store_true')
    args = ap.parse_args()

    with open(args.scenes, encoding='utf-8') as f:
        scenes = json.load(f)['scenes']
    todo = [s for s in scenes if s.get('sign')]

    import easyocr
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=not args.no_gpu, verbose=False)
    print('[plate] easyocr ready', flush=True)

    items = []
    for sc in todo:
        sid = sc['sid']
        exp = sc['sign']['text']
        path = os.path.join(args.plate_dir, f'{sid}.png')
        if not os.path.exists(path):
            items.append({'sid': sid, 'expected': exp, 'verdict': 'MISSING'})
            continue
        with Image.open(path) as im:
            plate = im.convert('RGB')
        # 2 倍放大提升 OCR 命中率
        big = plate.resize((plate.width * 2, plate.height * 2), Image.LANCZOS)
        arr = np.array(big)
        try:
            res = reader.readtext(arr, detail=1, paragraph=False)
        except Exception:
            res = []
        got = [norm(t) for _, t, _ in res if norm(t)]
        best, best_s = '', 0.0
        for g in got:
            s = sim(g, exp)
            if s > best_s:
                best, best_s = g, s
        exact = any(g == norm(exp) for g in got)
        if exact:
            v = 'PASS'
        elif best_s >= 0.6:
            v = 'NEAR'
        else:
            v = 'FAIL'
        items.append({'sid': sid, 'expected': exp, 'got': got, 'best': best,
                      'sim': round(best_s, 3), 'verdict': v,
                      'style': sc['sign'].get('style'),
                      'plate_size': list(plate.size)})
        print(f"{v:5} {sid:26} exp={exp!r} got={got} sim={best_s:.2f}", flush=True)

    n_pass = sum(1 for r in items if r['verdict'] == 'PASS')
    n_near = sum(1 for r in items if r['verdict'] == 'NEAR')
    summary = {'total': len(items), 'pass': n_pass, 'near': n_near,
               'fail': len(items) - n_pass - n_near}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'items': items}, f, ensure_ascii=False, indent=1)
    print('=== summary ===', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
