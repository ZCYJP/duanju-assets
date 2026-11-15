#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把背景图拼成联系表（contact sheet），便于整体查看与抽检。

用法: build_sheets.py --dir /ssd2/duanju/scenes_b100_signed --out /ssd2/duanju/sheets \
                     --cols 5 --rows 5 --cell 220
"""
import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

SCENES = '/ssd2/duanju/scripts/scenes_b100.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scenes', default=SCENES)
    ap.add_argument('--cols', type=int, default=5)
    ap.add_argument('--rows', type=int, default=5)
    ap.add_argument('--cell', type=int, default=220)
    ap.add_argument('--label_h', type=int, default=26)
    args = ap.parse_args()

    with open(args.scenes, encoding='utf-8') as f:
        scenes = json.load(f)['scenes']
    order = [s['sid'] for s in scenes]

    files = [f for f in order if os.path.exists(os.path.join(args.dir, f'{f}.png'))]
    if not files:
        files = sorted(os.path.splitext(f)[0] for f in os.listdir(args.dir) if f.endswith('.png'))
    os.makedirs(args.out, exist_ok=True)

    per = args.cols * args.rows
    cw, ch, lh = args.cell, int(args.cell * 16 / 9), args.label_h
    font = None
    for fp in ('/usr/share/fonts/truetype/custom/SourceHanSansSC-Bold.otf',
               '/usr/share/fonts/truetype/custom/NotoSansSC-Bold.ttf'):
        if not os.path.exists(fp):
            continue
        try:
            font = ImageFont.truetype(fp, 15)
            break
        except OSError:
            continue

    n_sheet = 0
    for start in range(0, len(files), per):
        chunk = files[start:start + per]
        sheet = Image.new('RGB', (args.cols * cw, args.rows * (ch + lh)), (245, 245, 247))
        d = ImageDraw.Draw(sheet)
        for i, sid in enumerate(chunk):
            r, c = divmod(i, args.cols)
            x, y = c * cw, r * (ch + lh)
            try:
                with Image.open(os.path.join(args.dir, f'{sid}.png')) as im:
                    im = im.convert('RGB').resize((cw - 4, ch - 4), Image.LANCZOS)
                    sheet.paste(im, (x + 2, y + 2))
            except Exception as ex:
                if font is not None:
                    d.text((x + 6, y + 6), f'ERR {ex}'[:30], fill=(200, 0, 0), font=font)
            sc = next((s for s in scenes if s['sid'] == sid), None)
            label = sid.replace('bg100_', '')
            if sc and sc.get('sign'):
                label += f'  [{sc["sign"]["text"]}]'
            if font is not None:
                d.text((x + 5, y + ch + 2), label[:34], fill=(30, 30, 34), font=font)
        n_sheet += 1
        dst = os.path.join(args.out, f'sheet_{n_sheet}.png')
        sheet.save(dst, quality=92)
        print(f'{dst}  ({len(chunk)} 张)')


if __name__ == '__main__':
    main()
