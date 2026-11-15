#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把四层文字审计 scenes_100_text_final.json 转成 build_scenes_index.py 期望的报告格式。

旧报告 text_report_full.json 标称 100/100 OK，但判据过宽（例如把 "EXAACT\\n外卖驿站"
判为 OK），与四层审计（OCR + VL全图 + VL招牌 + 放大复核）实测的 91 真实 / 9 伪文字冲突。
本脚本以四层审计为准，产出 text_report_audit_v4.json。

输出项: sid, lang, expected, transcript, status
  status ∈ {OK, PSEUDO, NOTEXT}
"""
import json
import os
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "/ssd2/duanju/scenes_100_text_final.json"
DST = sys.argv[2] if len(sys.argv) > 2 else "/ssd2/duanju/scenes_100/text_report_audit_v4.json"


def transcript_of(item):
    """拼接各层实际读到的招牌文字，作为页面上的"识别"行。"""
    parts = []
    for key in ("ocr_best", "vl_best"):
        v = (item.get(key) or "").strip()
        if v and v not in parts:
            parts.append(v)
    for v in item.get("vl_lines") or []:
        v = (v or "").strip()
        if v and v not in parts:
            parts.append(v)
    if not parts:
        for v in item.get("signcheck_lines") or []:
            v = (v or "").strip()
            if v and v not in parts:
                parts.append(v)
    return " / ".join(parts)


def main():
    data = json.load(open(SRC, encoding="utf-8"))
    items = data["items"]
    out = []
    for it in items:
        verdict = it.get("verdict", "")
        if verdict == "真实文字":
            status = "OK"
        elif verdict == "伪文字":
            status = "PSEUDO"
        else:
            status = "NOTEXT"
        out.append({
            "sid": it["sid"],
            "lang": it.get("lang", ""),
            "expected": it.get("expected", ""),
            "transcript": transcript_of(it),
            "status": status,
            "confidence": it.get("confidence", ""),
            "reason": it.get("reason", ""),
            "pseudo_evidence": it.get("pseudo_evidence", []),
        })

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    from collections import Counter
    c = Counter(x["status"] for x in out)
    print("written", DST, len(out), "items", dict(c))


if __name__ == "__main__":
    main()
