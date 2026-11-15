#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建季节/节气变体任务池：150 角色 × 26 变体 = 3900 个任务。

按 (cid, variant) 交错排列，保证各 worker 负载均衡。
"""
import json, os, shutil, sys

sys.path.insert(0, '/ssd2/duanju/scripts')
from season_scenes import VARIANTS

POOL = sys.argv[1] if len(sys.argv) > 1 else '/ssd2/duanju/pool_season'
PEND = os.path.join(POOL, 'pending')
CAST = '/ssd2/duanju/scripts/cast_all_150.json'

if os.path.isdir(PEND):
    shutil.rmtree(PEND)
os.makedirs(PEND, exist_ok=True)

cast = json.load(open(CAST, encoding='utf-8'))
tasks = []
for c in cast:
    cid = c['id']
    sex_off = 0 if cid[0] == 'M' else 50
    base_seed = 1600000 + sex_off + int(cid[1:]) * 100
    scene_base = c.get('scene_en', 'in a softly lit neutral interior with shallow depth of field')
    for vi, (key, zh, en, mood, clothes) in enumerate(VARIANTS):
        tasks.append({
            'kind': 'season', 'cid': cid, 'variant': key, 'zh': zh,
            'en': en, 'mood': mood, 'clothes': clothes, 'scene_base': scene_base,
            'seed': base_seed + vi * 10,
        })

# 交错：先按 variant 轮转再按角色轮转，避免同一 worker 拿到同一角色的所有变体
tasks.sort(key=lambda t: (t['variant'], t['cid']))

for i, t in enumerate(tasks):
    name = f't_{i:05d}_{t["cid"]}_{t["variant"]}'
    with open(os.path.join(PEND, name + '.json'), 'w', encoding='utf-8') as f:
        json.dump(t, f, ensure_ascii=False)

print(f'season pool: {len(tasks)} tasks ({len(cast)} roles x {len(VARIANTS)} variants)')
print(f'pending: {PEND}')
