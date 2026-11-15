#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建共享任务池：100 张背景 + 100 个角色（各10视图），交错排列供多实例抢占。"""
import json, os, re, shutil, sys

POOL = sys.argv[1] if len(sys.argv) > 1 else '/ssd2/duanju/pool'
PEND = os.path.join(POOL, 'pending')

# 重建池目录（保留 done/failed 便于追溯）
for sub in ('pending',):
    d = os.path.join(POOL, sub)
    if os.path.isdir(d):
        shutil.rmtree(d)
os.makedirs(PEND, exist_ok=True)

# ---- 背景任务 ----
src = open('/ssd2/duanju/scripts/gen_scenes_100.py', encoding='utf-8').read()
ns = {}
m = re.search(r'SCENES = \[(.*?)\n\]', src, re.S)
exec('SCENES = [' + m.group(1) + '\n]', ns)
SCENES = ns['SCENES']
assert len(SCENES) == 100, len(SCENES)

scene_tasks = []
for i, (sid, cn, en, text, lang) in enumerate(SCENES):
    scene_tasks.append({
        'kind': 'scene', 'sid': sid, 'cn': cn, 'en': en, 'text': text, 'lang': lang,
        'seed': 770000 + i * 37,
    })

# ---- 角色任务（只取新增的 100 人）----
cast = json.load(open('/ssd2/duanju/scripts/cast_all_150.json', encoding='utf-8'))
new_cast = [c for c in cast if int(c['id'][1:]) >= 26]
assert len(new_cast) == 100, len(new_cast)

char_tasks = []
for c in new_cast:
    char_tasks.append({'kind': 'char', 'cid': c['id'], 'cast': c})

# ---- 交错排列：背景与角色轮流，让每张卡既画背景也画人 ----
tasks = []
maxn = max(len(scene_tasks), len(char_tasks))
for i in range(maxn):
    if i < len(scene_tasks):
        tasks.append(scene_tasks[i])
    if i < len(char_tasks):
        tasks.append(char_tasks[i])

for i, t in enumerate(tasks):
    name = f't_{i:04d}_' + (t['sid'] if t['kind'] == 'scene' else t['cid'])
    with open(os.path.join(PEND, name + '.json'), 'w', encoding='utf-8') as f:
        json.dump(t, f, ensure_ascii=False)

print(f'pool tasks: {len(tasks)} ({len(scene_tasks)} scenes + {len(char_tasks)} chars)')
print(f'pending dir: {PEND}')
