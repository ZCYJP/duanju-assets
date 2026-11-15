#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从核验日志重建完整的背景文字核验报告。

背景
----
/ssd2/duanju/scenes_100/text_report_v2.json 每跑一次就被覆盖，只保留当次
参与核验的条目（最后一次只有 2 条），无法反映 100 张的全貌。
但真实的核验输出全部留在日志里：
  textcheck_all.log    —— 首轮全量 100 张（98 通过）
  textcheck_9.log      —— 重生成 9 张后的复验
  textcheck_last2.log  —— 最后 2 张的复验
日志行格式： OK <sid> [<lang>] exp='<期望>' got='<识别到>'
失败行格式：FAIL <sid> ...

本脚本按 sid 去重合并（后跑的日志覆盖先跑的，因为是复验），
输出完整报告到 text_report_full.json。
"""
import glob
import json
import os
import re
import sys

LOG_DIR = sys.argv[1] if len(sys.argv) > 1 else "/ssd2/duanju/logs"
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    "/ssd2/duanju/scenes_100/text_report_full.json"

# 日志按时间先后排序：先跑的全量，再跑的重生成复验（后者应覆盖前者）
files = sorted(
    glob.glob(os.path.join(LOG_DIR, "textcheck*.log")),
    key=lambda p: os.path.getmtime(p))

LINE_OK = re.compile(
    r"^OK\s+(\S+)\s+\[([^\]]+)\]\s+exp='([^']*)'\s+got='(.*)'\s*$")
LINE_FAIL = re.compile(r"^FAIL\s+(\S+)\s+\[([^\]]+)\]\s+exp='([^']*)'\s*(.*)$")

report = {}
order = []
for fp in files:
    n_ok = n_fail = 0
    with open(fp, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            m = LINE_OK.match(line)
            if m:
                sid, lang, exp, got = m.groups()
                if sid not in report:
                    order.append(sid)
                report[sid] = {
                    "sid": sid, "lang": lang, "expected": exp,
                    "transcript": got, "status": "OK",
                }
                n_ok += 1
                continue
            m = LINE_FAIL.match(line)
            if m:
                sid, lang, exp, rest = m.groups()
                if sid not in report:
                    order.append(sid)
                report[sid] = {
                    "sid": sid, "lang": lang, "expected": exp,
                    "transcript": rest.strip(), "status": "MISMATCH",
                }
                n_fail += 1
    print(f"[parse] {os.path.basename(fp)}: ok={n_ok} fail={n_fail}")

items = [report[s] for s in order]
ok_n = sum(1 for r in items if r["status"] == "OK")
print(f"[merge] 合并后共 {len(items)} 条，OK={ok_n}，"
      f"FAIL={len(items)-ok_n}")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(items, f, ensure_ascii=False, indent=1)
print("[write]", OUT)
