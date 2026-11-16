#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并 12 分片 OCR 结果，按置信度分级，标出疑似伪字。"""
import glob
import json
import os
import sys

WORK = '/ssd2/duanju/scenes_jp200'
SCENES = json.load(open('/ssd2/duanju/scripts/scenes_jp200.json', encoding='utf-8'))['scenes']
by_sid = {s['sid']: s for s in SCENES}

merged = {}
for jf in sorted(glob.glob(os.path.join(WORK, 'ocr_shard*.json'))):
    for rec in json.load(open(jf, encoding='utf-8')):
        merged[rec['path'][:-4]] = rec

print(f'merged {len(merged)} / 200')

# 汇总每张图的最高置信命中
rows = []
for sid, rec in merged.items():
    sc = by_sid.get(sid, {})
    hits = rec.get('hits', [])
    confs = [h['conf'] for h in hits]
    maxc = max(confs) if confs else 0.0
    texts = [h['text'] for h in hits]
    rows.append((sid, sc.get('name', ''), bool(sc.get('sign')), len(hits), maxc, texts))

# 分级：高置信(>=0.5) 需人工核；中(0.35~0.5) 存疑；低(<0.35) 视为纹理误报
def verdict(sid, has_sign, maxc, texts):
    if maxc >= 0.5:
        return 'HIGH'
    if maxc >= 0.35:
        return 'MID'
    if maxc > 0:
        return 'LOW'
    return 'CLEAN'

rows.sort(key=lambda r: -r[4])
hi = [r for r in rows if verdict(*r[:1], r[2], r[4], r[5]) == 'HIGH']
mi = [r for r in rows if verdict(r[0], r[2], r[4], r[5]) == 'MID']
lo = [r for r in rows if verdict(r[0], r[2], r[4], r[5]) == 'LOW']
cl = [r for r in rows if verdict(r[0], r[2], r[4], r[5]) == 'CLEAN']

print(f'\n=== 分级 ===  HIGH(>=0.5):{len(hi)}  MID(0.35-0.5):{len(mi)}  LOW(<0.35):{len(lo)}  CLEAN:{len(cl)}')
print('\n--- HIGH（需人工核验）---')
for r in hi:
    print(f'  {r[0]:32s} {r[1]:14s} sign={int(r[2])} maxc={r[4]:.3f} {r[5]}')
print('\n--- MID（存疑）---')
for r in mi:
    print(f'  {r[0]:32s} {r[1]:14s} sign={int(r[2])} maxc={r[4]:.3f} {r[5]}')

json.dump({'high': hi, 'mid': mi, 'low': lo, 'clean': [r[0] for r in cl]},
          open(os.path.join(WORK, 'ocr_triage.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('\n-> ocr_triage.json')
