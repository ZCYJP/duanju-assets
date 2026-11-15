#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""背景图汉字真实性校验。

L1 原图扫描：贴字前对整图 OCR，检出模型自绘的任何文字（这些是伪字风险源）
L2 牌面回读：贴字后裁剪牌面区域 OCR，与预期文案逐字比对
L3 对比度  ：牌面与其周边背景的亮度差，防止字贴上去看不见

判定：
  回读完全一致           -> PASS
  相似度 >= 0.8          -> PASS(近似)
  其它                   -> FAIL

用法:
  verify_text.py --scenes scenes_b100.json --raw /ssd2/duanju/scenes_b100 \
                  --signed /ssd2/duanju/scenes_b100_signed --out /ssd2/duanju/bg100_verify.json
"""
import argparse
import difflib
import json
import os
import re

os.environ.setdefault('OMP_THREAD_LIMIT', '1')

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

READERS = {}


def norm(s):
    if not s:
        return ''
    s = s.strip()
    s = re.sub(r'\s+', '', s)
    s = s.replace('：', ':').replace('，', ',').replace('。', '.')
    s = ''.join(ch for ch in s if ch.isprintable())
    return s


def sim(a, b):
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def get_reader(gpu=True):
    global READERS
    if not READERS:
        import easyocr
        READERS['sim'] = easyocr.Reader(['ch_sim', 'en'], gpu=gpu, verbose=False)
    return READERS['sim']


def ocr_full(img, reader):
    arr = np.array(img.convert('RGB'))
    try:
        res = reader.readtext(arr, detail=1, paragraph=False)
    except Exception:
        return []
    out = []
    for box, txt, conf in res:
        txt = norm(txt)
        if not txt:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        out.append({'text': txt, 'conf': round(float(conf), 3),
                    'box': [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]})
    return out


def crop_sign(img, box, pad_ratio=0.12):
    W, H = img.size
    x0, y0, x1, y1 = box
    bw, bh = (x1 - x0) * W, (y1 - y0) * H
    px, py = bw * pad_ratio, bh * pad_ratio
    l = max(0, int(x0 * W - px))
    t = max(0, int(y0 * H - py))
    r = min(W, int(x1 * W + px))
    b = min(H, int(y1 * H + py))
    return img.crop((l, t, r, b))


def contrast(img, box):
    """牌面区域与周边一圈背景的亮度差（0-255）。"""
    W, H = img.size
    x0, y0, x1, y1 = box
    l, t, r, b = int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)
    face = np.array(img.convert('L').crop((l, t, r, b)), dtype=float)
    if face.size == 0:
        return 0.0
    ow = int((r - l) * 0.6) or 4
    oh = int((b - t) * 0.6) or 4
    surround = []
    if l - ow >= 0:
        surround.append(np.array(img.convert('L').crop((l - ow, t, l, b)), dtype=float))
    if r + ow <= W:
        surround.append(np.array(img.convert('L').crop((r, t, r + ow, b)), dtype=float))
    if t - oh >= 0:
        surround.append(np.array(img.convert('L').crop((l, t - oh, r, t)), dtype=float))
    if b + oh <= H:
        surround.append(np.array(img.convert('L').crop((l, b, r, b + oh)), dtype=float))
    if not surround:
        return 255.0
    bg = np.concatenate([s.ravel() for s in surround])
    return float(abs(face.mean() - bg.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', required=True)
    ap.add_argument('--raw', required=True)
    ap.add_argument('--signed', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--no_gpu', action='store_true')
    ap.add_argument('--only', default='')
    args = ap.parse_args()

    with open(args.scenes, encoding='utf-8') as f:
        scenes = json.load(f)['scenes']
    if args.only:
        want = set(args.only.split(','))
        scenes = [s for s in scenes if s['sid'] in want]

    reader = get_reader(gpu=not args.no_gpu)
    print('[verify] easyocr ready', flush=True)

    items = []
    for sc in scenes:
        sid = sc['sid']
        sign = sc.get('sign')
        raw_path = os.path.join(args.raw, f'{sid}.png')
        signed_path = os.path.join(args.signed, f'{sid}.png')
        rec = {'sid': sid, 'group': sc.get('group'), 'expected': sign['text'] if sign else None,
               'style': sign.get('style') if sign else None}
        if not os.path.exists(raw_path):
            rec['status'] = 'MISSING_RAW'
            items.append(rec)
            continue
        if not sign:
            rec['status'] = 'NO_SIGN'
            items.append(rec)
            continue
        if not os.path.exists(signed_path):
            rec['status'] = 'MISSING_SIGNED'
            items.append(rec)
            continue

        with Image.open(raw_path) as im:
            raw = im.convert('RGB')
        with Image.open(signed_path) as im:
            signed = im.convert('RGB')

        # L1 原图是否被模型画了字
        raw_hits = ocr_full(raw, reader)
        raw_txt = [h['text'] for h in raw_hits]
        rec['raw_ocr'] = raw_txt

        # L2 牌面回读
        crop = crop_sign(signed, sign['box'])
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        hits = ocr_full(crop, reader)
        got = [h['text'] for h in hits]
        best, best_s = '', 0.0
        for g in got:
            s = sim(g, sign['text'])
            if s > best_s:
                best, best_s = g, s
        exact = any(norm(g) == norm(sign['text']) for g in got)
        rec['sign_ocr'] = got
        rec['sign_ocr_best'] = best
        rec['similarity'] = round(best_s, 4)
        rec['exact'] = exact

        # L3 对比度
        c = contrast(signed, sign['box'])
        rec['contrast'] = round(c, 1)

        if exact or best_s >= 0.8:
            rec['status'] = 'PASS'
        else:
            rec['status'] = 'FAIL'
        if c < 18:
            rec['warn'] = '牌面与背景对比度过低'
        if raw_txt:
            rec['warn_raw_text'] = '原图检出模型自绘文字，需确认是否为伪字'
        items.append(rec)
        print(f"{rec['status']:4} {sid:28} exp={sign['text']!r} got={got} sim={best_s:.2f} "
              f"contrast={c:.0f}", flush=True)

    n_pass = sum(1 for r in items if r.get('status') == 'PASS')
    n_sign = sum(1 for r in items if r.get('status') in ('PASS', 'FAIL'))
    n_raw = sum(1 for r in items if r.get('raw_ocr'))
    n_lowc = sum(1 for r in items if r.get('contrast', 999) < 18)
    summary = {
        'total': len(items),
        'with_sign': n_sign,
        'pass': n_pass,
        'fail': n_sign - n_pass,
        'raw_has_model_text': n_raw,
        'low_contrast': n_lowc,
    }
    out = {'summary': summary, 'items': items}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print('=== summary ===', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
