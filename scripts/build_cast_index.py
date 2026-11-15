#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the cast index page for the 16-character digital-human asset library.

usage: build_cast_index.py <cast.json> <images_root> <out.html>
"""
import json, os, sys, html

VIEWS = [
    ('hero', '主形象', '半身正面·电影感场景'),
    ('headshot', '面部特写', '纯色影棚·数字人换脸用'),
    ('threequarter', '3/4 侧', '头转向约40度'),
    ('profile', '正侧面', '90度侧脸·轮廓参考'),
    ('back', '背面', '全身背影视角'),
    ('fullbody', '全身', '头到脚完整全身'),
    ('turnaround', '三视图设定表', '正面/侧面/背面·横版'),
    ('smile', '微笑', '表情变体'),
    ('angry', '愤怒', '表情变体'),
    ('sad', '悲伤', '表情变体'),
]


def main():
    cast = json.load(open(sys.argv[1], encoding='utf-8'))
    root = sys.argv[2]
    out = sys.argv[3]
    voices_root = sys.argv[4] if len(sys.argv) > 4 else ''
    voices_meta = {}
    # 优先使用合并后的总 manifest（新旧两批音色）
    vman_candidates = []
    if voices_root:
        vman_candidates += [
            os.path.join(os.path.dirname(voices_root.rstrip('/')) if voices_root.endswith('/')
                         else os.path.dirname(voices_root), 'voice_manifest_all.json'),
            os.path.join(os.path.dirname(voices_root.rstrip('/')) if voices_root.endswith('/')
                         else os.path.dirname(voices_root), 'voice_manifest.json'),
        ]
    for vman in vman_candidates:
        if os.path.exists(vman):
            for it in json.load(open(vman, encoding='utf-8')).get('items', []):
                voices_meta[it.get('id')] = it
            break
    rel_root = os.path.relpath(root, os.path.dirname(os.path.abspath(out)))

    cards = []
    for c in cast:
        cid = c['id']
        tiles = []
        for key, zh, desc in VIEWS:
            p = os.path.join(root, cid, f'{cid}_{key}.png')
            if not os.path.exists(p):
                continue
            rel = os.path.relpath(p, os.path.dirname(os.path.abspath(out)))
            tiles.append(f'''<figure class="v">
  <a href="{html.escape(rel)}" target="_blank"><img src="{html.escape(rel)}" loading="lazy" alt="{cid} {zh}"></a>
  <figcaption><b>{zh}</b><span>{desc}</span></figcaption>
</figure>''')
        gender_cls = 'm' if c['gender'] == '男' else 'f'
        # 音色试听（若存在）
        voice_block = ''
        if voices_root:
            vs = []
            # 音色可能分布在 samples / samples_ext 两个目录（新旧两批角色）
            roots = [voices_root]
            if voices_root.rstrip('/').endswith('samples'):
                roots.append(voices_root.rstrip('/') + '_ext')
            for vr in roots:
                for ext in ('.wav', '.mp3'):
                    vp = os.path.join(vr, f"{cid}_{c['name']}_voice{ext}")
                    if os.path.exists(vp):
                        vs.append((vp, os.path.relpath(vp, os.path.dirname(os.path.abspath(out)))))
                        break
                if vs:
                    break
            if vs:
                vp, rel = vs[0]
                vm = voices_meta.get(cid, {})
                vdesc = vm.get('voice_desc') or vm.get('desc', '')
                vf0 = vm.get('f0', '')
                vdur = vm.get('dur', '')
                voice_block = f"""<div class="voice">
  <audio controls preload="none" src="{html.escape(rel)}"></audio>
  <span class="vdesc">音色：{html.escape(vdesc)}</span>
  <span class="vf0">基频 F0≈{vf0}Hz{(' · 时长 ' + str(vdur) + 's') if vdur else ''}</span>
</div>"""
        cards.append(f'''<section class="card {gender_cls}">
  <header>
    <div class="who"><span class="badge">{cid}</span><h2>{html.escape(c['name'])}</h2>
      <span class="tag">{c['gender']} · {c['age']}岁</span></div>
    <p class="role">{html.escape(c['role'])}</p>
  </header>
  <dl class="meta">
    <div><dt>长相</dt><dd>{html.escape(c['look'])}</dd></div>
    <div><dt>服装</dt><dd>{html.escape(c['outfit'])}</dd></div>
    <div><dt>场景</dt><dd>{html.escape(c['scene'])}</dd></div>
  </dl>
  <div class="views">{''.join(tiles)}</div>
  {voice_block}
</section>''')

    males = [c for c in cast if c['gender'] == '男']
    females = [c for c in cast if c['gender'] == '女']

    html_doc = f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>短剧数字人角色库 · {len(males)}男{len(females)}女</title>
<style>
:root{{--bg:#f7f7f8;--card:#fff;--ink:#1c1c1e;--sub:#6b6b70;--line:#e6e6ea;--m:#2f6fed;--f:#c2447e;--shadow:0 1px 3px rgba(0,0,0,.06),0 8px 24px rgba(0,0,0,.05);}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;}}
.wrap{{max-width:1400px;margin:0 auto;padding:32px 24px 80px;}}
header.top{{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:28px;}}
header.top h1{{margin:0 0 6px;font-size:26px;letter-spacing:-.3px;}}
header.top p{{margin:0;color:var(--sub);font-size:14px;}}
.stats{{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px;}}
.chip{{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:5px 14px;font-size:13px;color:var(--sub);}}
.chip b{{color:var(--ink);}}
.quicklinks{{margin-top:14px;}}
.quicklinks a{{display:inline-block;background:var(--ink);color:#fff;text-decoration:none;padding:8px 16px;border-radius:8px;font-size:13px;font-weight:600;}}
.quicklinks a:hover{{opacity:.86;}}
h2.sec{{font-size:18px;margin:36px 0 16px;padding-left:11px;border-left:3px solid var(--ink);}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);padding:22px;margin-bottom:22px;}}
.card>header{{border-bottom:1px solid var(--line);padding-bottom:14px;margin-bottom:14px;}}
.who{{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;}}
.badge{{font:600 12px/1 ui-monospace,SFMono-Regular,Menlo,monospace;background:#f0f0f3;color:var(--sub);padding:5px 8px;border-radius:6px;}}
.card h2{{margin:0;font-size:20px;}}
.tag{{font-size:13px;color:var(--sub);}}
.role{{margin:8px 0 0;color:var(--m);font-weight:600;font-size:14px;}}
.card.f .role{{color:var(--f);}}
dl.meta{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px 22px;margin:0 0 16px;}}
dl.meta div{{display:flex;gap:8px;font-size:13px;}}
dl.meta dt{{color:var(--sub);flex:0 0 34px;}}
dl.meta dd{{margin:0;color:#3a3a3f;}}
.voice{{margin-top:14px;padding:12px 14px;background:#f7f7f9;border:1px solid var(--line);border-radius:10px;display:flex;flex-direction:column;gap:5px;}}
.voice audio{{height:32px;max-width:420px;}}
.vdesc{{font-size:12px;color:#3a3a3f;line-height:1.55;}}
.vf0{{font-size:11px;color:var(--sub);}}
.views{{display:grid;grid-template-columns:repeat(auto-fill,minmax(158px,1fr));gap:12px;}}
.v{{margin:0;background:#fafafb;border:1px solid var(--line);border-radius:10px;overflow:hidden;}}
.v img{{width:100%;display:block;aspect-ratio:3/4;object-fit:cover;background:#eee;}}
.v:has(img[src$="turnaround.png"]) img{{aspect-ratio:16/9;}}
.v figcaption{{padding:8px 10px;display:flex;flex-direction:column;gap:2px;}}
.v figcaption b{{font-size:13px;}}
.v figcaption span{{font-size:11px;color:var(--sub);}}
footer{{margin-top:44px;padding-top:18px;border-top:1px solid var(--line);color:var(--sub);font-size:12px;}}
</style></head><body><div class="wrap">
<header class="top">
<h1>短剧数字人角色库 · {len(males)} 男 {len(females)} 女</h1>
<p>每位角色 10 个视图：主形象 / 面部特写 / 3-4侧 / 正侧面 / 背面 / 全身 / 三视图设定表 / 微笑 / 愤怒 / 悲伤。全部基于同一张脸派生，可直接用于选角、分镜参考与数字人驱动。</p>
<div class="stats">
<span class="chip">角色 <b>{len(cast)}</b> 人</span>
<span class="chip">男 <b>{len(males)}</b> · 女 <b>{len(females)}</b></span>
<span class="chip">视图 <b>{len(cast)*len(VIEWS)}</b> 张</span>
<span class="chip">生成 <b>Qwen-Image-Edit-2511 + Lightning 4步</b></span>
<span class="chip">一致性校验 <b>InsightFace + VL 双检</b></span>
  <span class="chip">音色 <b>Qwen3-TTS VoiceDesign</b> 每角色独立</span>
</div>
<div class="quicklinks">
<a href="{html.escape(os.path.relpath(os.path.join(root, '..', 'sheets', 'all_heroes.jpg'), os.path.dirname(os.path.abspath(out))))}" target="_blank">{len(cast)} 人主形象总览大图 →</a>
</div>
</header>
<h2 class="sec">男性角色（{len(males)}）</h2>
{''.join(card for card, c in zip(cards, cast) if c['gender']=='男')}
<h2 class="sec">女性角色（{len(females)}）</h2>
{''.join(card for card, c in zip(cards, cast) if c['gender']=='女')}
<footer>图片点击可查看原图，音频可试听。所有资产由 h100-038 节点生成，服务器路径 /ssd2/duanju/</footer>
</div></body></html>'''

    with open(out, 'w', encoding='utf-8') as f:
        f.write(html_doc)
    print('written', out, len(html_doc), 'bytes')


if __name__ == '__main__':
    main()
