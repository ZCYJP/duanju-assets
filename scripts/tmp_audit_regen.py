#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 scenes_100_regen/ 下重生成的 9 张背景跑完整文字审计。

复用 tmp_bg_text_audit.py 的全部判据与函数（OCR + tesseract 交叉 + 文字区域检测 +
裁剪放大二次 OCR + Unicode 字符集规则 + 预期文本语义匹配），只把 ROOT / CROP_DIR /
OUT_JSON 换成重生成目录，并补上 sid 白名单。

用法: python3 tmp_audit_regen.py [root] [out_json]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tmp_bg_text_audit as A

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/ssd2/duanju/scenes_100_regen"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/ssd2/duanju/scenes_100_regen_text_audit.json"

A.ROOT = ROOT
A.CROP_DIR = os.path.join(ROOT, "_crops")
A.OUT_JSON = OUT

# EXPECTED 由 main() 里的 load_expected() 填充；本脚本不走 main，需显式调用，
# 否则 audit_one 取不到预期文本，LANG_TO_READER 查不到语言而 KeyError 'sim'。
A.EXPECTED = A.load_expected()
print(f"[init] EXPECTED entries = {len(A.EXPECTED)}", flush=True)

# easyocr reader 是懒加载缓存，必须显式初始化，否则 READERS 为空 -> KeyError 'sim'
A.init_readers(use_gpu=True)


def main():
    # 重生成的目标：四层审计判定为伪文字的 9 张
    src = "/ssd2/duanju/scenes_100_text_final.json"
    data = json.load(open(src, encoding="utf-8"))
    targets = [x["sid"] for x in data["items"] if x.get("verdict") == "伪文字"]

    files = []
    for sid in targets:
        p = os.path.join(ROOT, f"{sid}.png")
        if os.path.exists(p):
            files.append(p)
        else:
            print(f"[skip] missing {p}", flush=True)

    os.makedirs(A.CROP_DIR, exist_ok=True)
    print(f"[audit] {len(files)} files under {ROOT}", flush=True)

    results = []
    for i, p in enumerate(files):
        try:
            r = A.audit_one(p)
            results.append(r)
            print(f"  [{i+1}/{len(files)}] {os.path.basename(p)} -> "
                  f"{r.get('verdict')} conf={r.get('confidence')} "
                  f"ocr={r.get('ocr_best')} vl={r.get('vl_best')}", flush=True)
        except Exception as ex:
            print(f"  [err] {p}: {ex}", flush=True)

    # 汇总
    from collections import Counter
    c = Counter(r.get("verdict") for r in results)
    out = {
        "meta": {
            "root": ROOT,
            "crop_dir": A.CROP_DIR,
            "targets": targets,
            "n_audited": len(results),
            "note": "复用 tmp_bg_text_audit.py 全部判据，仅改根目录",
        },
        "summary": dict(c),
        "items": results,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[done] -> {OUT}  {dict(c)}", flush=True)


if __name__ == "__main__":
    main()
