#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""季节/节气变体 + 9 情绪音色的索引页。

usage: build_season_index.py <cast.json> <seasons_root> <emotions_root> <out.html>
"""
import html
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from season_scenes import VARIANTS

# 9 种情绪定义（与 gen_emotion_voices.py 一致）
EMOTIONS = [
    ("xi", 1, "喜", "音调偏高、节奏轻快跳跃、富有弹性", "清脆笑声、欢呼"),
    ("nu", 2, "怒", "音量极大、重音明显、语速急促有爆发力", "怒吼、咆哮、咬牙低哼"),
    ("ai", 3, "哀", "音调低沉、音量微弱、语速缓慢带颤抖", "啜泣、抽泣、绝望低语"),
    ("le", 4, "乐", "音调柔和平稳、音量适中、节奏舒缓松弛", "满足轻哼、轻松呼吸"),
    ("ju", 5, "惧", "音调发尖或发哑、呼吸紊乱、明显颤抖", "尖叫、倒吸凉气、结巴"),
    ("jing", 6, "惊", "极具突发性、短促吸气、音调瞬间大幅上扬", "惊呼、倒吸冷气"),
    ("yan", 7, "厌", "发声偏鼻腔或喉咙后部、带摩擦感、语速慢", "撇嘴、鼻腔冷哼"),
    ("you", 8, "忧", "音量偏轻、节奏不均（停顿多）、气息绵长", "欲言又止、长叹气"),
    ("xiu", 9, "羞", "音量极小、气声重、小心翼翼、犹豫停顿", "尴尬干笑、吸鼻子"),
]


def main():
    cast = json.load(open(sys.argv[1], encoding="utf-8"))
    seasons_root = sys.argv[2]
    emo_root = sys.argv[3]
    out = sys.argv[4]
    outdir = os.path.dirname(os.path.abspath(out))
    rel = lambda p: os.path.relpath(p, outdir)

    blocks = []
    for c in cast:
        cid = c["id"]
        sdir = os.path.join(seasons_root, cid)
        edir = os.path.join(emo_root, cid)

        tiles = []
        for key, zh, en, mood, clothes in VARIANTS:
            p = os.path.join(sdir, f"{cid}_{key}.png")
            if not os.path.exists(p):
                continue
            tiles.append(f'''<figure class="v">
  <a href="{html.escape(rel(p))}" target="_blank"><img src="{html.escape(rel(p))}" loading="lazy" alt="{cid} {zh}"></a>
  <figcaption><b>{zh}</b><span>{html.escape(mood)}</span></figcaption>
</figure>''')

        voices = []
        for ek, eno, ezh, feat, typ in EMOTIONS:
            p = os.path.join(edir, f"{cid}_{ek}.wav")
            if not os.path.exists(p):
                continue
            voices.append(f'''<div class="emo">
  <span class="eno">{eno}</span><span class="ezh">{ezh}</span>
  <audio controls preload="none" src="{html.escape(rel(p))}"></audio>
  <span class="efeat">{html.escape(feat)}</span>
  <span class="etyp">典型：{html.escape(typ)}</span>
</div>''')

        gender_cls = 'm' if c['gender'] == '男' else 'f'
        blocks.append(f'''<section class="card {gender_cls}">
  <header>
    <div class="who"><span class="badge">{cid}</span><h2>{html.escape(c['name'])}</h2>
      <span class="tag">{c['gender']} · {c['age']}岁</span></div>
    <p class="role">{html.escape(c['role'])}</p>
  </header>
  <h3 class="sub">季节 / 24 节气形象（{len(tiles)} 张）</h3>
  <div class="views">{''.join(tiles)}</div>
  <h3 class="sub">9 种情绪音色（{len(voices)} 条）</h3>
  <div class="emos">{''.join(voices)}</div>
</section>''')

    males = [c for c in cast if c['gender'] == '男']
    females = [c for c in cast if c['gender'] == '女']

    doc = f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>短剧数字人 · 季节节气与情绪音色库 · {len(cast)}人</title>
<style>
:root{{--bg:#f7f7f8;--card:#fff;--ink:#1c1c1e;--sub:#6b6b70;--line:#e6e6ea;--m:#2f6fed;--f:#c2447e;--shadow:0 1px 3px rgba(0,0,0,.06),0 8px 24px rgba(0,0,0,.05);}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;}}
.wrap{{max-width:1500px;margin:0 auto;padding:32px 24px 80px;}}
header.top{{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:28px;}}
header.top h1{{margin:0 0 6px;font-size:26px;letter-spacing:-.3px;}}
header.top p{{margin:0;color:var(--sub);font-size:14px;}}
.stats{{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px;}}
.chip{{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:5px 14px;font-size:13px;color:var(--sub);}}
.chip b{{color:var(--ink);}}
h2.sec{{font-size:18px;margin:36px 0 16px;padding-left:11px;border-left:3px solid var(--ink);}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);padding:22px;margin-bottom:22px;}}
.card>header{{border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:14px;}}
.who{{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;}}
.badge{{font:600 12px/1 ui-monospace,SFMono-Regular,Menlo,monospace;background:#f0f0f3;color:var(--sub);padding:5px 8px;border-radius:6px;}}
.card h2{{margin:0;font-size:20px;}}
.tag{{font-size:13px;color:var(--sub);}}
.role{{margin:6px 0 0;color:var(--m);font-weight:600;font-size:14px;}}
.card.f .role{{color:var(--f);}}
h3.sub{{font-size:14px;color:var(--sub);margin:16px 0 10px;font-weight:600;}}
.views{{display:grid;grid-template-columns:repeat(auto-fill,minmax(148px,1fr));gap:10px;}}
.v{{margin:0;background:#fafafb;border:1px solid var(--line);border-radius:10px;overflow:hidden;}}
.v img{{width:100%;display:block;aspect-ratio:3/4;object-fit:cover;background:#eee;}}
.v figcaption{{padding:7px 9px;display:flex;flex-direction:column;gap:1px;}}
.v figcaption b{{font-size:13px;}}
.v figcaption span{{font-size:10px;color:var(--sub);line-height:1.4;}}
.emos{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px;}}
.emo{{background:#f7f7f9;border:1px solid var(--line);border-radius:10px;padding:10px 12px;display:flex;flex-direction:column;gap:3px;}}
.emo audio{{height:30px;width:100%;}}
.eno{{display:inline-block;background:var(--ink);color:#fff;font-size:11px;width:18px;height:18px;line-height:18px;text-align:center;border-radius:50%;margin-right:5px;}}
.ezh{{font-weight:700;font-size:14px;}}
.efeat{{font-size:12px;color:#3a3a3f;}}
.etyp{{font-size:11px;color:var(--sub);}}
footer{{margin-top:44px;padding-top:18px;border-top:1px solid var(--line);color:var(--sub);font-size:12px;}}
</style></head><body><div class="wrap">
<header class="top">
<h1>短剧数字人 · 季节节气与情绪音色库</h1>
<p>每位角色在冬、夏与二十四节气共 26 个时令场景下的形象变体，以及喜 / 怒 / 哀 / 乐 / 惧 / 惊 / 厌 / 忧 / 羞 九种情绪音色。全部形象以角色主形象为同一张脸派生，音色在角色原声线基础上叠加情绪特征。</p>
<div class="stats">
<span class="chip">角色 <b>{len(cast)}</b> 人（男 {len(males)} · 女 {len(females)}）</span>
<span class="chip">时令形象 <b>{len(cast)*26}</b> 张</span>
<span class="chip">情绪音色 <b>{len(cast)*9}</b> 条</span>
<span class="chip">形象模型 <b>Qwen-Image-Edit-2511 + Lightning 4步</b></span>
<span class="chip">音色模型 <b>Qwen3-TTS VoiceDesign</b></span>
</div>
</header>
<h2 class="sec">男性角色（{len(males)}）</h2>
{''.join(b for b, c in zip(blocks, cast) if c['gender']=='男')}
<h2 class="sec">女性角色（{len(females)}）</h2>
{''.join(b for b, c in zip(blocks, cast) if c['gender']=='女')}
<footer>图片点击查看原图，音频可试听。资产由 h100-038 节点生成，服务器路径 /ssd2/duanju/</footer>
</div></body></html>'''

    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    print("written", out, len(doc), "bytes")


if __name__ == "__main__":
    main()
