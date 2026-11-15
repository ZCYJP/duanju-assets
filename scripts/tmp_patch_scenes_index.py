#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 build_scenes_index.py 改为优先读取四层审计报告 text_report_audit_v4.json。

背景：
- 旧报告 text_report_full.json 标称 100/100 OK，但判据过宽
  （例如 bg_delivery_street 的 "EXAACT\\n外卖驿站" 被判为 OK）。
- 四层审计（OCR + VL全图 + VL招牌 + 放大复核）实测 91 真实 / 9 伪文字。
- 索引页必须反映真实核验结果，并区分 OK / PSEUDO 两种状态。
"""
import io
import re
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "/ssd2/duanju/scripts/build_scenes_index.py"
src = io.open(PATH, encoding="utf-8").read()

OLD_LOAD = '''    report = {}
    for name in ("text_report_full.json", "text_report_v2.json"):
        rp = os.path.join(scenes_root, name)
        if os.path.exists(rp):
            for r in json.load(open(rp, encoding="utf-8")):
                report[r["sid"]] = r
            break
'''

NEW_LOAD = '''    # 优先使用四层审计报告（OCR + VL全图 + VL招牌 + 放大复核）。
    # 旧报告 text_report_full.json 标称 100/100 OK，但判据过宽，不能作为依据。
    report = {}
    for name in ("text_report_audit_v4.json", "text_report_full.json", "text_report_v2.json"):
        rp = os.path.join(scenes_root, name)
        if os.path.exists(rp):
            for r in json.load(open(rp, encoding="utf-8")):
                report[r["sid"]] = r
            break
'''

OLD_BADGE = '''            st = r.get("status", "")
            ok = "OK" if st == "OK" else ("MISSING" if st == "MISSING" else "待核")
            badge = "ok" if ok == "OK" else "warn"
            got = r.get("transcript", "")
            gotline = (f'<span class="got">识别：{html.escape(got)}</span>'
                       if got and got.strip() else "")
'''

NEW_BADGE = '''            st = r.get("status", "")
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
'''

assert OLD_LOAD in src, "load block not found"
src = src.replace(OLD_LOAD, NEW_LOAD)

assert OLD_BADGE in src, "badge block not found"
src = src.replace(OLD_BADGE, NEW_BADGE)

OLD_CARD = "{gotline}\n</figcaption>\n</figure>''')\n"

NEW_CARD = "{gotline}\n{pseudoline}\n</figcaption>\n</figure>''')\n"

assert OLD_CARD in src, "card block not found"
src = src.replace(OLD_CARD, NEW_CARD)

# 统计：真实文字数 / 伪文字数
OLD_STAT = '''    ok_n = sum(1 for r in report.values() if r.get("status") == "OK")
    total = len(scenes)
    shown = sum(len(v) for v in by_lang.values())
'''

NEW_STAT = '''    ok_n = sum(1 for r in report.values() if r.get("status") == "OK")
    pseudo_n = sum(1 for r in report.values() if r.get("status") == "PSEUDO")
    notext_n = sum(1 for r in report.values() if r.get("status") == "NOTEXT")
    total = len(scenes)
    shown = sum(len(v) for v in by_lang.values())
'''

assert OLD_STAT in src, "stat block not found"
src = src.replace(OLD_STAT, NEW_STAT)

OLD_CHIP = '''<span class="chip">文字核验通过 <b>{ok_n}</b> / {shown}</span>
'''

NEW_CHIP = '''<span class="chip">真实文字 <b>{ok_n}</b> / {shown}</span>
<span class="chip">伪文字 <b>{pseudo_n}</b> 张</span>
<span class="chip">核验方式 <b>OCR + 视觉模型 + 放大复核</b></span>
'''

assert OLD_CHIP in src, "chip block not found"
src = src.replace(OLD_CHIP, NEW_CHIP)

# 样式：新增 .badge.bad 与 .pseudo
OLD_CSS = '''.badge.warn{{background:#fdf1e3;color:var(--warn);}}
'''

NEW_CSS = '''.badge.warn{{background:#fdf1e3;color:var(--warn);}}
.badge.bad{{background:#fdeaea;color:#c0392b;}}
.pseudo{{display:block;font-size:11px;color:#c0392b;margin-top:4px;font-weight:600;}}
'''

assert OLD_CSS in src, "css block not found"
src = src.replace(OLD_CSS, NEW_CSS)

# 文案：不再声称"均为真实汉字"
OLD_DESC = '''<p>100 张用于短剧的场景背景，涵盖城市街道、店铺内景、交通枢纽、办公场所、自然景观等。每张招牌上的文字均为真实汉字或可拼读的外文，已由视觉模型逐张核验。</p>'''

NEW_DESC = '''<p>100 张用于短剧的场景背景，涵盖城市街道、店铺内景、交通枢纽、办公场所、自然景观等。招牌文字经 OCR、视觉模型全图与招牌区域判读、放大复核四层核验；标注为"伪文字"的图片含 AI 生成的乱码字符，仅作占位用途。</p>'''

assert OLD_DESC in src, "desc block not found"
src = src.replace(OLD_DESC, NEW_DESC)

io.open(PATH, "w", encoding="utf-8").write(src)
print("patched", PATH)
