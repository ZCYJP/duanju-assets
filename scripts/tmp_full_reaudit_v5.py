#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用修正后的文字规则 v5，重跑 100 张原始背景 + 9 张重生成背景的完整四层核验。

四层：
  L1 OCR 字符集硬规则   （v5 规则：空白 + 拉丁扩展合法）
  L2 VL 整图转写
  L3 VL 逐招牌真实/伪标注
  L4 VL 放大逐字符 + 拉丁语义判定
  最终裁决（四层融合）

用法:
  python tmp_full_reaudit_v5.py orig     # 100 张原始
  python tmp_full_reaudit_v5.py regen     # 9 张重生成
"""
import json
import os
import subprocess
import sys

HERE = "/ssd2/duanju/scripts"
D = "/ssd2/duanju"
PY = "/ssd2/envs/qwentts/bin/python"

REGEN_SIDS = ["bg_auto_shop", "bg_delivery_street", "bg_driving_office",
              "bg_grain_depot", "bg_home_services", "bg_kindergarten_paint",
              "bg_law_office", "bg_pet_shop", "bg_tofu_stall"]


def sh(cmd, env=None, timeout=3000):
    e = dict(os.environ)
    e["CUDA_VISIBLE_DEVICES"] = "1"
    if env:
        e.update(env)
    print(f"$ {cmd[:140]}", flush=True)
    r = subprocess.run(cmd, env=e, cwd=HERE, shell=isinstance(cmd, str),
                       timeout=timeout)
    return r.returncode


def run_l1(root, crops, out_json, sids=None):
    """L1: OCR 全量审计（v5 规则）。"""
    code = f'''
import sys, os, json
sys.path.insert(0, {HERE!r})
import tmp_text_rule_v5 as R
A = R.apply()
A.ROOT = {root!r}
A.CROP_DIR = {crops!r}
A.OUT_JSON = {out_json!r}
A.EXPECTED = A.load_expected()
A.init_readers(use_gpu=True)
import glob
if {sids!r} is None:
    files = sorted(glob.glob(os.path.join(A.ROOT, "*.png")))
else:
    files = [os.path.join(A.ROOT, s + ".png") for s in {sids!r}]
    files = [f for f in files if os.path.exists(f)]
os.makedirs(A.CROP_DIR, exist_ok=True)
results = []
for i, p in enumerate(files):
    try:
        rec = A.audit_one(p)
    except Exception as ex:
        import traceback; traceback.print_exc()
        rec = {{"file": os.path.basename(p), "path": p, "verdict": "无文字",
               "reason": f"处理异常: {{ex}}", "ocr_strings": [], "crops": [],
               "n_text_regions": 0, "similarity": 0.0, "expected": "",
               "pseudo_strings": [], "real_strings": [], "text_regions": [],
               "matched": False, "best_match": ""}}
    results.append(rec)
    print(f"  [{{i+1}}/{{len(files)}}] {{rec['file']}} -> {{rec['verdict']}}", flush=True)
from collections import Counter
c = Counter(r.get("verdict") for r in results)
summary = {{"total": len(results)}}
for k in ("真实文字", "伪文字", "无文字"):
    summary[k] = sum(1 for r in results if r.get("verdict") == k)
summary["伪文字图片"] = [r["file"] for r in results if r.get("verdict") == "伪文字"]
summary["无文字图片"] = [r["file"] for r in results if r.get("verdict") == "无文字"]
out = {{"meta": {{"scenes_dir": A.ROOT, "crop_dir": A.CROP_DIR,
                 "rule": "v5: 空白 + ASCII字母数字 + 拉丁扩展 + CJK/假名/谚文",
                 "ocr_engines": ["easyocr 1.7.2 (GPU)", "tesseract 4.1.1"]}},
       "summary": summary, "items": results}}
json.dump(out, open(A.OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("[L1 done]", summary, flush=True)
'''
    return sh([PY, "-c", code])


def run_layer(mod, root, crops, out_path, sids=None, vl_json=None):
    code = f'''
import os, sys
sys.path.insert(0, {HERE!r})
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import {mod} as M
M.ROOT = {root!r}
M.CROP_DIR = {crops!r}
M.OUT = {out_path!r}
if hasattr(M, "VL_JSON"):
    M.VL_JSON = {vl_json or out_path!r}
sys.argv = ["{mod}"] + {sids!r}
M.main()
'''
    return sh([PY, "-c", code])


def run_adjudicate(ocr, vl, sc, zm, sem, out):
    code = f'''
import sys
sys.path.insert(0, {HERE!r})
import tmp_text_rule_v5 as R
J = R.apply_adjudicate()
J.OCR_JSON = {ocr!r}
J.VL_JSON  = {vl!r}
J.SC_JSON  = {sc!r}
J.ZM_JSON  = {zm!r}
J.SEM_JSON = {sem!r}
J.OUT      = {out!r}
J.main()
'''
    return sh([PY, "-c", code])


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "regen"

    if which == "orig":
        root = f"{D}/scenes_100"
        crops = f"{D}/bg_text_crops_v5"
        sids = None
        pfx = "scenes_100"
    else:
        root = f"{D}/scenes_100_regen"
        crops = f"{root}/_crops_v5"
        sids = REGEN_SIDS
        pfx = "scenes_100_regen"

    ocr = f"{D}/{pfx}_text_audit_v5.json"
    vl = f"{D}/{pfx}_text_vl_verify_v5.json"
    sc = f"{D}/{pfx}_text_signcheck_v5.json"
    zm = f"{D}/{pfx}_text_zoomcheck_v5.json"
    sem = f"{D}/{pfx}_text_semantic_v5.json"
    fin = f"{D}/{pfx}_text_final_v5.json"

    print(f"=== {which}: {root} ===", flush=True)
    if run_l1(root, crops, ocr, sids) != 0:
        print("[L1 FAILED]", flush=True)
        return 1
    if run_layer("tmp_vl_verify", root, crops, vl, sids) != 0:
        print("[L2 FAILED]", flush=True)
        return 1
    if run_layer("tmp_vl_signcheck", root, crops, sc, sids) != 0:
        print("[L3 FAILED]", flush=True)
        return 1
    if run_layer("tmp_vl_zoom", root, crops, zm, sids) != 0:
        print("[L4a FAILED]", flush=True)
        return 1
    if run_layer("tmp_vl_semantic", root, crops, sem, vl_json=vl) != 0:
        print("[L4b FAILED]", flush=True)
        return 1
    if run_adjudicate(ocr, vl, sc, zm, sem, fin) != 0:
        print("[adjudicate FAILED]", flush=True)
        return 1

    d = json.load(open(fin, encoding="utf-8"))
    print("=== FINAL ===", flush=True)
    print(json.dumps(d.get("summary", {}), ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
