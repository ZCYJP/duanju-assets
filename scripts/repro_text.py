#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小复现：找出 PIL 绘制多字时只画出一个字的条件。"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = '/usr/share/fonts/truetype/custom/SourceHanSansSC-Bold.otf'
TEXT = '白鹿书院'
FG = (238, 204, 132)
BG = (104, 30, 28)
SIZE = 66
W, H = 313, 165


def stat(img, label, fg=FG):
    arr = np.asarray(img.convert('RGBA'), dtype=np.int16)
    d = np.abs(arr[:, :, :3] - np.array(fg, dtype=np.int16)).sum(axis=2)
    hit = (d < 60) & (arr[:, :, 3] > 128)
    ys, xs = np.nonzero(hit)
    bbox = f'({xs.min()},{ys.min()})-({xs.max()},{ys.max()})' if hit.sum() else '无'
    print(f'{label:52} 字色像素={int(hit.sum()):6} bbox={bbox}')


def variant(label, rgba, fill4, stroke, stroke_fill, use_bbox, fill=FG):
    img = Image.new('RGBA' if rgba else 'RGB', (W, H), BG + ((255,) if rgba else ()))
    d = ImageDraw.Draw(img)
    f = ImageFont.truetype(FONT, SIZE)
    if use_bbox:
        bb = f.getbbox(TEXT)
        x = (W - (bb[2] - bb[0])) / 2 - bb[0]
        y = (H - (bb[3] - bb[1])) / 2 - bb[1]
    else:
        x, y = 20, 40
    kw = {}
    if stroke is not None:
        kw['stroke_width'] = stroke
    if stroke_fill is not None:
        kw['stroke_fill'] = stroke_fill
    d.text((x, y), TEXT, font=f, fill=fill + ((255,) if fill4 else ()), **kw)
    stat(img, label)
    return img


variant('A RGBA fill4 stroke0 stroke_fillNone +bbox', True, True, 0, None, True)
variant('B RGBA fill4 不传stroke +bbox', True, True, None, None, True)
variant('C RGB  fill3 stroke0 stroke_fillNone +bbox', False, False, 0, None, True)
variant('D RGBA fill4 stroke1 +bbox', True, True, 1, (20, 18, 16, 220), True)
variant('E RGBA fill4 stroke0 stroke_fillNone 固定xy', True, True, 0, None, False)

# 逐字绘制对照
img = Image.new('RGBA', (W, H), BG + (255,))
d = ImageDraw.Draw(img)
f = ImageFont.truetype(FONT, SIZE)
x = 24
for ch in TEXT:
    d.text((x, 40), ch, font=f, fill=FG + (255,))
    x += SIZE
stat(img, 'F RGBA 逐字绘制')

# 用 getmask2 渲染再 paste
img = Image.new('RGBA', (W, H), BG + (255,))
f = ImageFont.truetype(FONT, SIZE)
mask, offset = f.getmask2(TEXT, mode='L')
print(f'   getmask2 mask size={mask.size} offset={offset}')
lay = Image.new('RGBA', mask.size, FG + (255,))
lay.putalpha(mask)
img.paste(lay, (24 + offset[0], 40 + offset[1]), mask)
stat(img, 'G RGBA getmask2 + paste')
