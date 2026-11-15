#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定向重生成 bg_boutique（预期招牌 "精品店"）。

v5 四层裁决发现：主招牌 "精品店" 三个字完全正确（L1 OCR sim=1.00、L2 VL 读到
"精品店"、L3 标注"真实字符"），但 L4 放大层在旁支小字区域检出 "FJAA | [FAKE]"
—— 画面其他位置有 AI 生成的乱码字符。

做法：强化提示词，明确除主招牌外整个画面不允许出现任何字形
（包括小标签、价签、衣架标牌、包装字样），生成 4 个候选后逐张跑四层核验。

用法: python3 tmp_regen_boutique.py [--cands 4]
"""
import argparse
import json
import os
import sys
import time

HERE = "/ssd2/duanju/scripts"
sys.path.insert(0, HERE)

import gen_scenes_100 as G

OUT = "/ssd2/duanju/scenes_100_regen_cand"
SID = "bg_boutique"
TEXT = "精品店"

PROMPT = (
    "A photorealistic cinematic interior of an elegant boutique clothing store, "
    "empty of people, warm spotlights on a wooden display wall, "
    "mannequins and neatly folded garments on clean shelves. "
    f"A single plain illuminated sign above the entrance displays exactly three "
    f"Chinese characters: \"{TEXT}\". "
    "The three characters are rendered crisply and correctly "
    "(精-品-店), each glyph complete, legible and properly formed. "
    "Every other surface in the image is completely blank and clean: "
    "no clothing tags, no price tags, no shelf labels, no hang tags, "
    "no brand names on garments, no posters, no receipts, no business cards, "
    "no watermark, no digits, no random letters, no garbled or distorted glyphs. "
    "Only that one sign above the entrance carries text. "
    + G.STYLE
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    log = open(f"{OUT}/{SID}_regen.log", "a", encoding="utf-8")
    cands = []

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(f"=== {SID}: 生成 {args.cands} 个候选 ===")
    for i in range(args.cands):
        p = f"{OUT}/{SID}_b{i}.png"
        if os.path.exists(p) and os.path.getsize(p) > 20000:
            cands.append(p)
            say(f"  b{i} exists")
            continue
        seed = 440000 + 787 * i + 37
        for attempt in range(3):
            try:
                fs = G.run_one(G.up("blank:1216x832"), PROMPT,
                               seed + attempt * 31, f"{SID}_b{i}", OUT, G.NEG)
                if fs:
                    os.replace(fs[0], p)
                    cands.append(p)
                    say(f"  b{i} -> {p} attempt={attempt}")
                    break
            except Exception as ex:
                say(f"  b{i} attempt {attempt} failed: {ex}")
                time.sleep(3)

    with open(f"{OUT}/{SID}_cands.json", "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False, indent=1)
    say(f"=== {len(cands)} 个候选就绪 ===")


if __name__ == "__main__":
    main()
