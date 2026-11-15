#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 bg_driving_office 候选做 OCR 字符集校验（独立于生成步骤）。

生成在 /usr/bin/python3（需 PIL），easyocr 只在 /ssd2/envs/qwentts/bin/python 里，
故拆成两步：本脚本只负责校验已生成的候选。

用法: python tmp_ocr_cands.py <cand_dir>
"""
import json
import os
import re
import sys

CAND_DIR = sys.argv[1] if len(sys.argv) > 1 else "/ssd2/duanju/scenes_100_regen_cand"
SID = "bg_driving_office"
TEXT = "DRIVE SAFE"

ALLOWED = re.compile(r"^[A-Za-z0-9 ]*$")


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import easyocr

    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    files = sorted(f for f in os.listdir(CAND_DIR)
                   if f.startswith(SID) and f.endswith(".png"))
    print(f"[ocr] {len(files)} candidates in {CAND_DIR}", flush=True)

    results = []
    for f in files:
        p = os.path.join(CAND_DIR, f)
        try:
            res = reader.readtext(p, detail=1, paragraph=False)
            strs = [(r[1], float(r[2])) for r in res]
        except Exception as ex:
            strs = []
            print(f"  {f}: OCR error {ex}", flush=True)

        pseudo = False
        why = ""
        for t, _ in strs:
            s = (t or "").strip()
            if s and not ALLOWED.match(s):
                pseudo = True
                why = s
                break
        joined = " | ".join(t for t, _ in strs)
        match = TEXT.replace(" ", "") in joined.replace(" ", "").upper()
        results.append({"file": p, "pseudo": pseudo, "why": why,
                        "ocr": joined, "match": match})
        flag = "BAD " if pseudo else ("OK  " if match else "MISS")
        print(f"  {flag} {f}: ocr={joined!r} why={why!r}", flush=True)

    out = os.path.join(CAND_DIR, f"{SID}_ocr_candidates.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=1)

    good = [r for r in results if not r["pseudo"] and r["match"]]
    print(f"[done] good={len(good)}/{len(results)} -> {out}", flush=True)
    for r in good:
        print(f"  GOOD {os.path.basename(r['file'])}", flush=True)


if __name__ == "__main__":
    main()
