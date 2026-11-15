#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成日本角色资产索引页 japanese_index.html。

用法：build_japanese_index.py <cast.json> <images_root> <voices_root> <out.html>
out 必须放在仓库根目录，链接路径按相对 out 计算。
"""
import json, os, sys, html

VIEWS = [
    ('hero', '主形象', '全身正面·电影感场景'),
    ('headshot', '头部特写', '影棚·换表情用'),
    ('threequarter', '3/4 侧', '头转向约40度'),
    ('profile', '正侧面', '90度侧脸·轮廓参考'),
    ('back', '背面', '全身背影视角'),
    ('fullbody', '全身', '头到脚完整全身'),
    ('turnaround', '三视图设定表', '正面/侧面/背面·横版'),
    ('smile', '喜', '表情变体'),
    ('angry', '怒', '表情变体'),
    ('sad', '哀', '表情变体'),
]

EMOS = [('xi', '喜'), ('nu', '怒'), ('ai', '哀'), ('le', '乐'), ('ju', '惧'),
        ('jing', '惊'), ('yan', '厌'), ('you', '忧'), ('xiu', '羞')]

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--line:#e3e6ea;--txt:#1c2024;--sub:#6b7280;--accent:#2f6f4f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font-family:-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
header{padding:28px 32px 16px;border-bottom:1px solid var(--line);background:var(--card)}
h1{margin:0 0 6px;font-size:22px}
.meta{color:var(--sub);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:18px;padding:22px 32px 60px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
.card h2{margin:0;padding:12px 14px;font-size:16px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;align-items:baseline}
.card h2 small{color:var(--sub);font-weight:400;font-size:12px}
.tags{padding:8px 14px 0;color:var(--sub);font-size:12px}
.hero{padding:10px 14px}
.hero img{width:100%;border-radius:8px;display:block;background:#eee}
.views{display:grid;grid-template-columns:repeat(5,1fr);gap:6px;padding:0 14px 10px}
.views figure{margin:0}
.views img{width:100%;border-radius:6px;display:block;background:#eee;aspect-ratio:1/1;object-fit:cover}
.views figcaption{font-size:10px;color:var(--sub);text-align:center;margin-top:2px}
.voices{padding:6px 14px 14px}
.voices .row{display:flex;align-items:center;gap:8px;margin:6px 0}
.voices b{font-size:12px;width:18px;color:var(--accent)}
.voices audio{flex:1;height:30px}
"""

PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>日本角色资产库 · 100 个角色 · 多角度形象与 9 状态音色</title>
<style>{css}</style></head>
<body>
<header><h1>日本角色资产库 · 100 个角色 · 多角度形象与 9 状态音色</h1>
<div class="meta">每个角色 10 张形象图（主形象 / 8 视图 / 三视图设定表）+ 9 种状态音色（喜 / 怒 / 哀 / 乐 / 惧 / 惊 / 厌 / 忧 / 羞）· 共 {count} 种</div></header>
<div class="grid">{cards}</div>
</body></html>
"""


def main():
    animals_json, images_root, voices_root, out = sys.argv[1:5]
    raw = json.load(open(animals_json, encoding='utf-8'))
    animals = raw['animals'] if isinstance(raw, dict) else raw
    outdir = os.path.dirname(os.path.abspath(out))

    cards = []
    for a in animals:
        cid = a['id']
        tiles = []
        for key, zh, desc in VIEWS:
            p = os.path.join(images_root, cid, f'{cid}_{key}.png')
            if not os.path.exists(p):
                continue
            rel = os.path.relpath(p, outdir)
            tiles.append(f'<figure><a href="{html.escape(rel)}" target="_blank">'
                         f'<img src="{html.escape(rel)}" loading="lazy" alt="{cid} {zh}"></a>'
                         f'<figcaption>{zh}</figcaption></figure>')
        hero = os.path.join(images_root, cid, f'{cid}_hero.png')
        hero_rel = os.path.relpath(hero, outdir) if os.path.exists(hero) else ''
        vrows = []
        for ek, ezh in EMOS:
            p = os.path.join(voices_root, cid, f'{cid}_{ek}.wav')
            if not os.path.exists(p):
                continue
            rel = os.path.relpath(p, outdir)
            vrows.append(f'<div class="row"><b>{ezh}</b>'
                         f'<audio controls preload="none" src="{html.escape(rel)}"></audio></div>')
        cards.append(f'''<article class="card">
<h2>{html.escape(a.get('role', a['name']))} <small>{cid} · {a.get('age','')}岁 {html.escape(a.get('gender',''))}</small></h2>
<div class="tags">{html.escape(a.get('role',''))} · {a.get('age','')}岁 · {html.escape(a.get('gender',''))}</div>
<div class="hero"><a href="{html.escape(hero_rel)}" target="_blank"><img src="{html.escape(hero_rel)}" loading="lazy" alt="{cid} hero"></a></div>
<div class="views">{''.join(tiles)}</div>
<div class="voices">{''.join(vrows)}</div>
</article>''')

    html_out = PAGE.format(css=CSS, count=len(animals), cards='\n'.join(cards))
    with open(out, 'w', encoding='utf-8') as f:
        f.write(html_out)
    print(f'wrote {out} ({len(animals)} animals)')


if __name__ == '__main__':
    main()
