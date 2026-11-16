#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""精炼伪字嫌疑：剔除符号噪声、排除自有招牌文字，裁图放大成交付对照表。

判据：
- 命中文本至少含一个 CJK/假名
- conf >= 阈值（默认 0.5）
- 若场景带自有招牌 T：命中所有汉字/假名都在 T 字集内 -> 属"预期"（我们的真招牌），排除
- 其余 = 嫌疑（模型自绘伪字）
"""
import glob
import json
import os
import re
import unicodedata

from PIL import Image, ImageDraw, ImageFont

BASE = '/Users/baidu/WorkBuddy/H100/bg100'
IMGS = os.path.join(BASE, 'jp200_final')
OCR = os.path.join(BASE, 'jp200_ocr')
OUT = os.path.join(BASE, 'jp200_deliver')
os.makedirs(OUT, exist_ok=True)

CONF = 0.5
CJK = re.compile(r'[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff66-\uff9f]')


def has_cjk(t):
    return bool(CJK.search(t))


def chars(t):
    return set(c for c in t if CJK.match(c))


# 合并分片
merged = {}
for jf in sorted(glob.glob(os.path.join(OCR, 'ocr_shard*.json'))):
    for rec in json.load(open(jf, encoding='utf-8')):
        merged[rec['path'][:-4]] = rec

scenes = {s['sid']: s for s in json.load(open(os.path.join(BASE, 'scenes_jp200.json'), encoding='utf-8'))['scenes']}

# 视图 -> 全图坐标偏移
def to_full(view, x, y, W, H):
    if view == 'bottom':
        return x, y + H // 2
    if view == 'right':
        return x + W // 2, y
    return x, y


suspects = []   # (sid, text, conf, (cx0,cy0,cx1,cy1))
expected_sign = {}  # sid -> 命中的自有招牌确认
for sid, rec in merged.items():
    sc = scenes.get(sid, {})
    sign = sc.get('sign')
    sign_chars = chars(sign['text']) if sign else set()
    im = Image.open(os.path.join(IMGS, f'{sid}.png'))
    W, H = im.size
    for vn, items in rec['views'].items():
        for it in items:
            if 'error' in it:
                continue
            t, c = it['text'], it['conf']
            if c < CONF or not has_cjk(t):
                continue
            cc = chars(t)
            if sign and cc and cc <= sign_chars:
                expected_sign.setdefault(sid, []).append(t)
                continue
            xs = [p[0] for p in it['bbox']]
            ys = [p[1] for p in it['bbox']]
            fx0, fy0 = to_full(vn, min(xs), min(ys), W, H)
            fx1, fy1 = to_full(vn, max(xs), max(ys), W, H)
            suspects.append((sid, t, c, (fx0, fy0, fx1, fy1), vn))

# 去重：同 sid 同文本只留最高 conf / 最完整 bbox
dedup = {}
for sid, t, c, box, vn in suspects:
    k = (sid, t)
    if k not in dedup or c > dedup[k][1]:
        dedup[k] = (sid, c, box, vn)
suspects = [(s, t, c, box, vn) for (s, t), (_, c, box, vn) in dedup.items()]
suspects.sort(key=lambda r: (0 if not scenes.get(r[0], {}).get('sign') else 1, -r[2]))

print(f'suspects(去符号/去自有招牌) = {len(suspects)}')
print(f'scenes with 自有招牌confirmed: {len(expected_sign)}')
for sid, t, c, box, vn in suspects:
    tag = 'STREET' if scenes.get(sid, {}).get('sign') or sid.startswith('jpstr') else 'NATURE'
    print(f'  [{tag}] {sid:30s} conf={c:.3f} 「{t}」 {vn} {box}')

json.dump({'suspects': [{'sid': s, 'text': t, 'conf': c, 'box': box, 'view': vn}
                        for s, t, c, box, vn in suspects],
           'expected_sign': expected_sign},
          open(os.path.join(OCR, 'suspects.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)

# ---- 裁图对照表 ----
if not suspects:
    print('无嫌疑，跳过裁图')
    raise SystemExit
CW = 300
tiles = []
try:
    lab = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', 20)
    lab2 = ImageFont.truetype('/System/Library/Fonts/PingFang.ttc', 22)
except Exception:
    lab = lab2 = ImageFont.load_default()

for sid, t, c, (x0, y0, x1, y1), vn in suspects:
    im = Image.open(os.path.join(IMGS, f'{sid}.png')).convert('RGB')
    mx, my = max(14, (x1 - x0) * 0.6), max(14, (y1 - y0) * 0.8)
    cx0, cy0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
    cx1, cy1 = int(min(im.width, x1 + mx)), int(min(im.height, y1 + my))
    crop = im.crop((cx0, cy0, cx1, cy1))
    s = CW / crop.width
    crop = crop.resize((CW, max(1, int(crop.height * s))), Image.LANCZOS)
    tile = Image.new('RGB', (CW, crop.height + 52), (18, 20, 24))
    tile.paste(crop, (0, 0))
    d = ImageDraw.Draw(tile)
    d.text((4, crop.height + 4), f'{sid}', font=lab, fill=(240, 240, 240))
    d.text((4, crop.height + 26), f'「{t}」 c={c:.2f} {vn}', font=lab, fill=(224, 87, 74))
    tiles.append((sid, tile))

COLS = 6
GAP = 10
rows = (len(tiles) + COLS - 1) // COLS
TH = max(t[1].height for t in tiles)
W = COLS * CW + (COLS + 1) * GAP
H = rows * (TH + GAP) + GAP
sheet = Image.new('RGB', (W, H), (10, 11, 14))
for i, (sid, tile) in enumerate(tiles):
    r, c = divmod(i, COLS)
    sheet.paste(tile, (GAP + c * (CW + GAP), GAP + r * (TH + GAP)))
sp = os.path.join(OUT, 'suspects.png')
sheet.save(sp)
print('suspects sheet ->', sp, sheet.size)
