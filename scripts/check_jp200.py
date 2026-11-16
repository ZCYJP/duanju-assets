#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核验 jp200 产物：数量、可解码性、尺寸是否符合方向、是否有半截/0字节。"""
import json
import os
import sys

from PIL import Image

OUT = sys.argv[1] if len(sys.argv) > 1 else '/ssd2/duanju/scenes_jp200'
SCENES = '/ssd2/duanju/scripts/scenes_jp200.json'

scenes = {s['sid']: s for s in json.load(open(SCENES, encoding='utf-8'))['scenes']}
EXPECT = {'portrait': (752, 1392), 'landscape': (1392, 752)}

bad, missing, ok = [], [], 0
for sid, sc in scenes.items():
    p = os.path.join(OUT, f'{sid}.png')
    if not os.path.exists(p):
        missing.append(sid)
        continue
    if os.path.getsize(p) < 20000:
        bad.append((sid, 'small', os.path.getsize(p)))
        continue
    try:
        with Image.open(p) as im:
            im.load()
            sz = im.size
    except Exception as e:
        bad.append((sid, 'decode', str(e)))
        continue
    exp = EXPECT[sc['orientation']]
    if sz != exp:
        bad.append((sid, f'dim {sz} != {exp}'))
        continue
    ok += 1

print(f'ok={ok} missing={len(missing)} bad={len(bad)}')
if missing:
    print('MISSING:', missing)
if bad:
    for b in bad:
        print('BAD:', b)
