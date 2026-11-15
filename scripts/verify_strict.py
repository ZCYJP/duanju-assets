#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""严格字形验证：用合成时完全相同的字体/字号/排布重绘期望文本，
与实际牌面做二值化形态学匹配（IoU + 归一化相关）。

这是"画面上的笔画 == 期望汉字"的程序化铁证，
不依赖 OCR/VL 的识别能力（后者对艺术字/纹理底识别率低）。

做法：
  1. 用 paste_text 的 _fit_font 得到与实际渲染完全相同的 font/size/lines
  2. 把期望文本画到纯色底，得到"参考墨迹"
  3. 从实际牌面提取"文字墨迹"（按前景色与背景色的亮度差阈值化）
  4. 逐字在参考墨迹位置上做模板匹配（cv2.matchTemplate）
  5. 输出每字的匹配分数 + 整句 IoU

用法:
  verify_strict.py --plate_dir /ssd2/duanju/_diag_plate \
                   --scenes /ssd2/duanju/scripts/scenes_b100.json \
                   --out /ssd2/duanju/bg100_strict.json
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paste_text import STYLES, _fit_font, _draw_text  # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None

# 每种底色的"文字像素"判定：文字色与底色的亮度差
STYLE_FG_BG = {k: (v.get('fg'), v.get('bg')) for k, v in STYLES.items()}


def text_mask_from_plate(plate, style):
    """从实际牌面提取文字墨迹二值图（1=文字）。"""
    fg, bg = STYLE_FG_BG.get(style, (None, None))
    arr = np.asarray(plate.convert('RGB'), dtype=np.float32)
    gray = arr.mean(axis=2)
    if fg is not None and bg is not None:
        fgl = float(np.mean(fg))
        bgl = float(np.mean(bg)) if bg else None
        if bgl is not None:
            # 文字像素 = 亮度更接近 fg 且偏离 bg
            d_fg = np.abs(gray - fgl)
            d_bg = np.abs(gray - bgl)
            mask = (d_fg < d_bg * 0.75).astype(np.uint8)
        else:
            # 无底色（wall/paint）：文字是浅色，取高亮
            mask = (gray > np.percentile(gray, 55)).astype(np.uint8)
    else:
        mask = (gray > np.percentile(gray, 60)).astype(np.uint8)
    return mask


def render_reference(text, vertical, W, H, style):
    """用与 paste_text 完全一致的参数重绘文本，返回墨迹二值图 + 文字颜色。"""
    st = STYLES.get(style, STYLES['plaque'])
    font, size, lines = _fit_font(text, W, H, vertical)
    canvas = Image.new('RGB', (W, H), st['bg'] if st.get('bg') else (0, 0, 0))
    d = ImageDraw.Draw(canvas)
    fill = st['fg']
    _draw_text(d, lines, font, W, H, fill + (255,), vertical,
               stroke=0)
    arr = np.asarray(canvas, dtype=np.float32)
    gray = arr.mean(axis=2)
    fgl = float(np.mean(fill))
    bgl = float(np.mean(st['bg'])) if st.get('bg') else 0.0
    d_fg = np.abs(gray - fgl)
    d_bg = np.abs(gray - bgl)
    mask = (d_fg < d_bg * 0.75).astype(np.uint8) if st.get('bg') else (gray > 100).astype(np.uint8)
    return mask, size, canvas


from PIL import ImageDraw  # noqa: E402


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def best_match_score(hay, needle):
    """在 hay 中找 needle 的最佳归一化相关分。"""
    if cv2 is None:
        return 0.0
    r = cv2.matchTemplate(hay.astype(np.float32), needle.astype(np.float32),
                          cv2.TM_CCOEFF_NORMED)
    return float(np.nanmax(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plate_dir', required=True)
    ap.add_argument('--scenes', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', default='')
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
        text = sign['text']
        vertical = bool(sign.get('vertical'))
        style = sign.get('style', 'plaque')

        with Image.open(path) as im:
            plate = im.convert('RGB')
        W, H = plate.size

        actual = text_mask_from_plate(plate, style)
        ref, size, ref_rgb = render_reference(text, vertical, W, H, style)

        iou_v = iou(actual > 0, ref > 0)
        corr = best_match_score(actual, ref)

        # 逐字匹配：把参考逐字切出来，在实际墨迹里找
        per_char = []
        if cv2 is not None:
            ref_img = ref_rgb
            for i, ch in enumerate(text):
                # 逐字位置：用字体单独量
                from PIL import ImageFont
                f = ImageFont.truetype('/usr/share/fonts/truetype/custom/SourceHanSansSC-Bold.otf', size)
                bb = f.getbbox(ch)
                cw = bb[2] - bb[0]
                if vertical:
                    lh = int(size * 1.08)
                    total = lh * len(text)
                    y0 = int((H - total) / 2 + i * lh)
                    x0 = int((W - cw) / 2 - bb[0])
                    sub = ref[y0:y0 + lh, max(0, x0):max(0, x0) + cw]
                else:
                    lh = int(size * 1.08)
                    bb_all = f.getbbox(text)
                    tw = bb_all[2] - bb_all[0]
                    x0 = int((W - tw) / 2 - bb_all[0])
                    adv = f.getlength(text[:i]) if hasattr(f, 'getlength') else cw * i
                    sub = ref[max(0, int((H - (bb_all[3]-bb_all[1])) / 2 + bb_all[1])):
                              max(0, int((H - (bb_all[3]-bb_all[1])) / 2 + bb_all[1])) + int((bb_all[3]-bb_all[1]) * 1.15),
                              int(x0 + adv):int(x0 + adv + cw * 1.15)]
                if sub.size == 0 or sub.shape[0] > actual.shape[0] or sub.shape[1] > actual.shape[1]:
                    per_char.append({'ch': ch, 'score': 0.0})
                    continue
                s = best_match_score(actual, sub)
                per_char.append({'ch': ch, 'score': round(s, 3)})

        best_pc = min((p['score'] for p in per_char), default=0.0)
        avg_pc = float(np.mean([p['score'] for p in per_char])) if per_char else 0.0

        rec = {'sid': sid, 'expected': text, 'style': style, 'vertical': vertical,
               'font_size': size, 'plate_size': [W, H],
               'iou': round(iou_v, 4), 'corr': round(corr, 4),
               'per_char': per_char,
               'min_char_score': round(best_pc, 3), 'avg_char_score': round(avg_pc, 3),
               'actual_ink_ratio': round(float((actual > 0).mean()), 4),
               'ref_ink_ratio': round(float((ref > 0).mean()), 4)}
        items.append(rec)
        print(f"{sid:26} {text!r:14} sz={size:3} iou={iou_v:.3f} corr={corr:.3f} "
              f"min_pc={best_pc:.2f} avg_pc={avg_pc:.2f} "
              f"ink={rec['actual_ink_ratio']:.3f}/{rec['ref_ink_ratio']:.3f}", flush=True)

    ok = [r for r in items if r['iou'] >= 0.5 and r['min_char_score'] >= 0.5]
    summary = {'total': len(items), 'ok': len(ok), 'fail': len(items) - len(ok),
               'fail_sids': [r['sid'] for r in items if not (r['iou'] >= 0.5 and r['min_char_score'] >= 0.5)]}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'items': items}, f, ensure_ascii=False, indent=1)
    print('=== summary ===', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
