#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""背景索引页：100 张剧情背景 + 招牌文字核验结果。

用法：python3 build_scenes_index.py <cast_json> <scenes_root> <out.html>
"""
import glob
import html
import json
import os
import sys


def main():
    cast_json = sys.argv[1]
    scenes_root = sys.argv[2]
    out = sys.argv[3]
    outdir = os.path.dirname(os.path.abspath(out))

    src = open(cast_json, encoding="utf-8").read()
    ns = {}
    import re
    m = re.search(r"SCENES = \[(.*?)\n\]", src, re.S)
    exec("SCENES = [" + m.group(1) + "\n]", ns)
    scenes = ns["SCENES"]

    # 文字核验报告
    # 注意：text_report_v2.json 每跑一次就被覆盖，只保留当次参与核验的条目
    #（最后一次仅 2 条），不能代表 100 张全貌。
    # 完整结果由 tmp_merge_text_reports.py 从核验日志合并而来，
    # 落在 text_report_full.json（100/100 OK）。
    # 优先使用四层审计报告（OCR + VL全图 + VL招牌 + 放大复核）。
    # 旧报告 text_report_full.json 标称 100/100 OK，但判据过宽，不能作为依据。
    report = {}
    for name in ("text_report_audit_v5.json", "text_report_audit_v4.json", "text_report_full.json", "text_report_v2.json"):
        rp = os.path.join(scenes_root, name)
        if os.path.exists(rp):
            for r in json.load(open(rp, encoding="utf-8")):
                report[r["sid"]] = r
            break

    # 按语言分组
    by_lang = {}
    for sid, cn, en, text, lang in scenes:
        by_lang.setdefault(lang, []).append((sid, cn, en, text))

    lang_order = ["简体中文", "繁体中文", "日语", "韩语", "英语", "法语"]
    langs = [l for l in lang_order if l in by_lang] + \
            [l for l in by_lang if l not in lang_order]

    def rel(p):
        return os.path.relpath(p, outdir)

    blocks = []
    for lang in langs:
        items = by_lang[lang]
        cards = []
        for sid, cn, en, text in items:
            p = os.path.join(scenes_root, f"{sid}.png")
            if not os.path.exists(p):
                continue
            r = report.get(sid, {})
            st = r.get("status", "")
            if st == "OK":
                ok, badge = "真实文字", "ok"
            elif st == "PSEUDO":
                ok, badge = "伪文字", "bad"
            elif st == "NOTEXT":
                ok, badge = "无文字", "warn"
            else:
                ok, badge = "待核", "warn"
            got = r.get("transcript", "")
            gotline = (f'<span class="got">识别：{html.escape(got)}</span>'
                       if got and got.strip() else "")
            pseudo = r.get("pseudo_evidence") or []
            pseudoline = ""
            if pseudo:
                ev = "、".join(str(x) for x in pseudo)
                pseudoline = (f'<span class="pseudo">伪文字片段：{html.escape(ev)}</span>')
            cards.append(f'''<figure class="s">
<a href="{html.escape(rel(p))}" target="_blank"><img src="{html.escape(rel(p))}" loading="lazy" alt="{html.escape(cn)}"></a>
<figcaption>
<div class="t"><b>{html.escape(text)}</b><span class="badge {badge}">{ok}</span></div>
<div class="cn">{html.escape(cn)}</div>
<div class="en">{html.escape(en)}</div>
{gotline}
{pseudoline}
</figcaption>
</figure>''')
        blocks.append(f'''<h2 class="sec">{html.escape(lang)} <span class="cnt">{len(cards)} 张</span></h2>
<div class="grid">{"".join(cards)}</div>''')

    ok_n = sum(1 for r in report.values() if r.get("status") == "OK")
    pseudo_n = sum(1 for r in report.values() if r.get("status") == "PSEUDO")
    notext_n = sum(1 for r in report.values() if r.get("status") == "NOTEXT")
    total = len(scenes)
    shown = sum(len(v) for v in by_lang.values())
    doc = f'''<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>短剧剧情背景库 · 100 张</title>
<style>
:root{{--bg:#f7f7f8;--card:#fff;--ink:#1c1c1e;--sub:#6b6b70;--line:#e6e6ea;--ok:#2f9e5f;--warn:#c2680a;--shadow:0 1px 3px rgba(0,0,0,.06),0 8px 24px rgba(0,0,0,.05);}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;}}
.wrap{{max-width:1600px;margin:0 auto;padding:32px 24px 80px;}}
header.top{{border-bottom:1px solid var(--line);padding-bottom:20px;margin-bottom:24px;}}
header.top h1{{margin:0 0 6px;font-size:26px;letter-spacing:-.3px;}}
header.top p{{margin:0;color:var(--sub);font-size:14px;}}
.stats{{display:flex;gap:10px;flex-wrap:wrap;margin-top:16px;}}
.chip{{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:5px 14px;font-size:13px;color:var(--sub);}}
.chip b{{color:var(--ink);}}
h2.sec{{font-size:18px;margin:32px 0 14px;padding-left:11px;border-left:3px solid var(--ink);display:flex;align-items:baseline;gap:10px;}}
.cnt{{font-size:13px;color:var(--sub);font-weight:400;}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px;}}
.s{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;box-shadow:var(--shadow);}}
.s img{{width:100%;display:block;aspect-ratio:1216/832;object-fit:cover;background:#eee;}}
.s figcaption{{padding:10px 12px;}}
.t{{display:flex;align-items:center;justify-content:space-between;gap:8px;}}
.t b{{font-size:16px;}}
.badge{{font-size:11px;padding:2px 7px;border-radius:999px;font-weight:600;white-space:nowrap;}}
.badge.ok{{background:#e7f6ec;color:var(--ok);}}
.badge.warn{{background:#fdf1e3;color:var(--warn);}}
.badge.bad{{background:#fdeaea;color:#c0392b;}}
.pseudo{{display:block;font-size:11px;color:#c0392b;margin-top:4px;font-weight:600;}}
.cn{{font-size:13px;color:var(--ink);margin-top:3px;}}
.en{{font-size:11px;color:var(--sub);line-height:1.45;}}
.got{{display:block;font-size:11px;color:var(--sub);margin-top:4px;font-style:italic;}}
footer{{margin-top:44px;padding-top:18px;border-top:1px solid var(--line);color:var(--sub);font-size:12px;}}
</style></head><body><div class="wrap">
<header class="top">
<h1>短剧剧情背景库</h1>
<p>100 张用于短剧的场景背景，涵盖城市街道、店铺内景、交通枢纽、办公场所、自然景观等。招牌文字经 OCR、视觉模型全图与招牌区域判读、放大复核四层核验；标注为"伪文字"的图片含 AI 生成的乱码字符，仅作占位用途。</p>
<div class="stats">
<span class="chip">背景 <b>{shown}</b> 张</span>
<span class="chip">真实文字 <b>{ok_n}</b> / {shown}</span>
<span class="chip">伪文字 <b>{pseudo_n}</b> 张</span>
<span class="chip">核验方式 <b>OCR + 视觉模型 + 放大复核</b></span>
<span class="chip">语言种类 <b>{len(langs)}</b>（简 / 繁 / 日 / 韩 / 英 / 法）</span>
<span class="chip">模型 <b>Qwen-Image-Edit-2511 + Lightning 4步</b></span>
</div>
</header>
{"".join(blocks)}
<footer>点击图片查看原图。资产由 h100-038 节点生成，服务器路径 /ssd2/duanju/scenes_100/</footer>
</div></body></html>'''

    with open(out, "w", encoding="utf-8") as f:
        f.write(doc)
    print("written", out, len(doc), "bytes", f"ok={ok_n}/{total}")


if __name__ == "__main__":
    main()
