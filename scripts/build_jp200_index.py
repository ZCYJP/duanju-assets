#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""jp200 索引页：200 张日本氛围场景（自然环境 + 街景，含真实日文招牌）。

用法：python3 build_jp200_index.py <scenes_jp200.json> <scenes_dir_name> <out.html>
  scenes_dir_name 为仓库内相对目录名（默认 scenes_jp200），img src 用它拼相对路径。
"""
import html
import json
import os
import sys


def main():
    jf = sys.argv[1]
    dname = sys.argv[2] if len(sys.argv) > 2 else 'scenes_jp200'
    out = sys.argv[3] if len(sys.argv) > 3 else 'scenes_jp200_index.html'

    scenes = json.load(open(jf, encoding='utf-8'))['scenes']
    nat = [s for s in scenes if s['sid'].startswith('jpnat')]
    st = [s for s in scenes if s['sid'].startswith('jpstr')]
    n_sign = sum(1 for s in scenes if s.get('sign'))
    n_p = sum(1 for s in scenes if s.get('orientation') == 'portrait')
    n_l = sum(1 for s in scenes if s.get('orientation') == 'landscape')

    def card(s):
        sid = s['sid']
        ori = '竖' if s.get('orientation') == 'portrait' else '横'
        sg = s.get('sign')
        badge = f'<span class="sign">招牌「{html.escape(sg["text"])}」</span>' if sg else ''
        return (f'<figure class="card">'
                f'<img loading="lazy" src="{dname}/{sid}.png" alt="{html.escape(s["name"])}">'
                f'<figcaption><b>{html.escape(s["name"])}</b>'
                f'<span class="sid">{sid}</span>'
                f'<span class="tag">{ori}</span>{badge}</figcaption></figure>')

    def grid(items):
        return '<div class="grid">' + ''.join(card(s) for s in items) + '</div>'

    tpl = f'''<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>日本氛围场景图集 · 200</title>
<style>
:root{{--bg:#14161a;--card:#1e2127;--line:#2c3038;--fg:#e6e8ec;--dim:#9aa1ad;--acc:#e0574a}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 -apple-system,"Hiragino Sans","PingFang SC",sans-serif}}
header{{position:sticky;top:0;z-index:9;background:rgba(20,22,26,.92);backdrop-filter:blur(8px);
 border-bottom:1px solid var(--line);padding:16px 24px}}
h1{{margin:0 0 6px;font-size:20px}}
.sub{{color:var(--dim);font-size:13px}}
.sub b{{color:var(--acc)}}
h2{{margin:28px 24px 12px;font-size:16px;border-left:3px solid var(--acc);padding-left:10px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px;padding:0 24px 32px}}
.card{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
.card img{{width:100%;height:220px;object-fit:contain;background:#0d0f12;display:block}}
figcaption{{padding:8px 10px;font-size:12px;color:var(--dim);display:flex;flex-wrap:wrap;gap:6px;align-items:center}}
figcaption b{{color:var(--fg);font-weight:600}}
.sid{{font-family:ui-monospace,monospace;font-size:11px;opacity:.7}}
.tag{{background:#2a2e36;border-radius:4px;padding:1px 6px;color:#cfd4db}}
.sign{{background:var(--acc);color:#fff;border-radius:4px;padding:1px 6px}}
</style></head><body>
<header><h1>日本氛围场景图集 · 200 张</h1>
<div class="sub">自然环境 <b>{len(nat)}</b> · 街景 <b>{len(st)}</b> ·
 竖屏 9:16 <b>{n_p}</b> · 横屏 16:9 <b>{n_l}</b> · 含真实日文招牌 <b>{n_sign}</b>（真字体后期合成，非模型生成）</div>
</header>
<h2>自然环境 · {len(nat)}</h2>
{grid(nat)}
<h2>街景 · {len(st)}</h2>
{grid(st)}
</body></html>'''
    open(out, 'w', encoding='utf-8').write(tpl)
    print('index ->', out, f'{os.path.getsize(out)/1e6:.2f} MB', f'nat={len(nat)} st={len(st)} sign={n_sign} p={n_p} l={n_l}')


if __name__ == '__main__':
    main()
