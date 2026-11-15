#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定向重生成 bg_driving_office（英文标牌 "DRIVE SAFE"）。

第一轮重生成后四层裁决仍判伪：招牌上出现 "AIY" 这类乱码。
英文标牌是 AI 出图最容易把字母画歪的类型，这里用更强约束 + 多 seed 候选，
逐张跑 OCR 字符集校验，挑出无乱码的一张。

用法: python3 tmp_regen_driving_office.py [--cands 6]
"""
import argparse
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_scenes_100 as G

OUT = "/ssd2/duanju/scenes_100_regen_cand"
SID = "bg_driving_office"
TEXT = "DRIVE SAFE"

# 允许字符集：ASCII 字母数字 + 空格
ALLOWED = re.compile(r"^[A-Za-z0-9 ]*$")

PROMPT = (
    "A photorealistic cinematic interior of a driving school office, "
    "empty of people, warm daylight through large windows, "
    "a clean white wall with a single framed rectangular sign hanging on it. "
    f"The sign displays exactly two English words: \"{TEXT}\". "
    "The two words are in plain bold sans-serif capital letters, "
    "crisp and perfectly legible, every letter drawn correctly "
    "(D-R-I-V-E space S-A-F-E). "
    "There is absolutely no other text in the entire image: "
    "no small print, no posters, no license plates, no papers, "
    "no watermark, no random letters, no meaningless glyphs, "
    "no mirror or distorted characters. "
    "Completely clean surfaces everywhere else. "
    + G.STYLE
)


def ocr_strings(path):
    """用 easyocr 提取所有文本串（GPU 1 空闲）。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import easyocr
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    res = reader.readtext(path, detail=1, paragraph=False)
    return [(r[1], float(r[2])) for r in res]


def has_pseudo(strings):
    """任一串含不允许字符即判伪。"""
    for t, _ in strings:
        s = (t or "").strip()
        if not s:
            continue
        if not ALLOWED.match(s):
            return True, s
    return False, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", type=int, default=6)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    log = open(f"{OUT}/{SID}_regen.log", "a", encoding="utf-8")

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(f"=== {SID}: 生成 {args.cands} 个候选 ===")
    cands = []
    for i in range(args.cands):
        p = f"{OUT}/{SID}_c{i}.png"
        if os.path.exists(p) and os.path.getsize(p) > 20000:
            cands.append(p)
            say(f"  c{i} exists")
            continue
        seed = 770000 + 1000 * i + 31
        for attempt in range(3):
            try:
                fs = G.run_one(G.up("blank:1216x832"), PROMPT,
                               seed + attempt * 17, f"{SID}_c{i}", OUT, G.NEG)
                if fs:
                    os.replace(fs[0], p)
                    cands.append(p)
                    say(f"  c{i} -> {p} attempt={attempt}")
                    break
            except Exception as ex:
                say(f"  c{i} attempt {attempt} failed: {ex}")
                time.sleep(3)

    say(f"=== OCR 校验 {len(cands)} 个候选 ===")
    results = []
    for p in cands:
        try:
            strs = ocr_strings(p)
            bad, why = has_pseudo(strs)
            joined = " | ".join(t for t, _ in strs)
            results.append({"file": p, "pseudo": bad, "why": why,
                            "ocr": joined,
                            "match": TEXT.replace(" ", "") in joined.replace(" ", "").upper()})
            say(f"  {os.path.basename(p)}: pseudo={bad} why={why!r} ocr={joined!r}")
        except Exception as ex:
            say(f"  {os.path.basename(p)} OCR error: {ex}")
            results.append({"file": p, "pseudo": True, "why": f"ocr_error: {ex}",
                            "ocr": "", "match": False})

    with open(f"{OUT}/{SID}_ocr_candidates.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    good = [r for r in results if not r["pseudo"] and r["match"]]
    say(f"=== 可用候选 {len(good)} / {len(results)} ===")
    for r in good:
        say(f"  GOOD {r['file']}  ocr={r['ocr']!r}")
    if not good:
        say("=== 无完全合格候选，需人工/VL 复核 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
