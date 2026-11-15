#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定向重生成 bg_fresh_aisle（预期招牌 "新鲜蔬菜 每日直供"）。

v5 规则下 L1 OCR 层判定为伪文字：OCR 读到 "颦醭" / "攬議" / "雛a" 等形近字，
最高覆盖相似度仅 0.12，属真乱码而非规则误伤。

做法：主招牌只放 "新鲜蔬菜" 四个大字（去掉副标题，减少 AI 画错概率），
背景其他位置完全无文字；生成多候选后逐张跑四层核验挑合格者。

用法: python3 tmp_regen_fresh_aisle.py [--cands 4]
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
SID = "bg_fresh_aisle"
TEXT = "新鲜蔬菜"

PROMPT = (
    "A photorealistic cinematic interior of a fresh produce market aisle, "
    "empty of people, wooden crates full of green vegetables and red tomatoes, "
    "warm morning light through a skylight, shallow depth of field. "
    f"A single large plain white signboard is mounted above the aisle displaying "
    f"exactly four Chinese characters: \"{TEXT}\". "
    "The four characters are rendered crisply and correctly "
    "(新-鲜-蔬-菜), each glyph complete, legible and properly formed on the board. "
    "There is no other writing anywhere: no subtitles, no price tags, no labels, "
    "no posters, no banners, no phone numbers, no watermark, no digits, "
    "no random letters, no garbled or distorted glyphs. "
    "Only that one signboard carries text. "
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
        p = f"{OUT}/{SID}_f{i}.png"
        if os.path.exists(p) and os.path.getsize(p) > 20000:
            cands.append(p)
            say(f"  f{i} exists")
            continue
        seed = 515000 + 613 * i + 71
        for attempt in range(3):
            try:
                fs = G.run_one(G.up("blank:1216x832"), PROMPT,
                               seed + attempt * 29, f"{SID}_f{i}", OUT, G.NEG)
                if fs:
                    os.replace(fs[0], p)
                    cands.append(p)
                    say(f"  f{i} -> {p} attempt={attempt}")
                    break
            except Exception as ex:
                say(f"  f{i} attempt {attempt} failed: {ex}")
                time.sleep(3)

    with open(f"{OUT}/{SID}_cands.json", "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False, indent=1)
    say(f"=== {len(cands)} 个候选就绪 ===")


if __name__ == "__main__":
    main()
