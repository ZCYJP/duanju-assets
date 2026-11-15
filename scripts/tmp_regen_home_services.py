#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定向重生成 bg_home_services（繁体招牌 "家政服務"，画面含大量乱码）。

第二轮重生成后 L3 仍读到 "SONGS / SONGSONS / 107IF / 106R0" 等乱码，
说明招牌之外的区域（电话号码牌、车牌、传单等）仍在生成乱码字符。

做法：提示词里明确要求画面中除主招牌外完全无文字（连数字都不出现），
并多生成候选，用四层核验挑合格的一张。

用法: python3 tmp_regen_home_services.py [--cands 4]
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
SID = "bg_home_services"
TEXT = "家政服務"

PROMPT = (
    "A photorealistic cinematic interior of a small domestic-service agency office, "
    "empty of people, clean and tidy, daylight from a window. "
    f"A single rectangular signboard hangs on the wall displaying exactly these four "
    f"traditional Chinese characters: \"{TEXT}\". "
    "The four characters are rendered crisply and correctly "
    "(家-政-服-務), each glyph complete and legible on a plain white board. "
    "Everywhere else in the image is completely blank and clean: "
    "no telephone numbers, no price lists, no posters, no flyers, no business cards, "
    "no license plates, no small print, no watermark, no extra lettering of any kind, "
    "no digits anywhere, no random symbols, no garbled glyphs. "
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
        p = f"{OUT}/{SID}_h{i}.png"
        if os.path.exists(p) and os.path.getsize(p) > 20000:
            cands.append(p)
            say(f"  h{i} exists")
            continue
        seed = 630000 + 977 * i + 53
        for attempt in range(3):
            try:
                fs = G.run_one(G.up("blank:1216x832"), PROMPT,
                               seed + attempt * 23, f"{SID}_h{i}", OUT, G.NEG)
                if fs:
                    os.replace(fs[0], p)
                    cands.append(p)
                    say(f"  h{i} -> {p} attempt={attempt}")
                    break
            except Exception as ex:
                say(f"  h{i} attempt {attempt} failed: {ex}")
                time.sleep(3)

    with open(f"{OUT}/{SID}_cands.json", "w", encoding="utf-8") as f:
        json.dump(cands, f, ensure_ascii=False, indent=1)
    say(f"=== {len(cands)} 个候选就绪，需逐张跑四层核验 ===")


if __name__ == "__main__":
    main()
