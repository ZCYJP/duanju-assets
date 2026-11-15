#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用候选挑选器：对某个 sid 的多个候选逐张跑四层核验，挑出完全合格的一张。

用法: python tmp_pick_cands.py <sid> <cand_prefix> [n]
  例: python tmp_pick_cands.py bg_fresh_aisle bg_fresh_aisle_f 4
      python tmp_pick_cands.py bg_home_services bg_home_services_h 4
"""
import json
import os
import shutil
import subprocess
import sys

HERE = "/ssd2/duanju/scripts"
D = "/ssd2/duanju"
PY = "/ssd2/envs/qwentts/bin/python"
CAND = f"{D}/scenes_100_regen_cand"


def sh(code, timeout=2400, cpu=False):
    e = dict(os.environ)
    e["CUDA_VISIBLE_DEVICES"] = "" if cpu else "1"
    print(f"$ {code.strip().splitlines()[0][:100]}", flush=True)
    return subprocess.run([PY, "-c", code], env=e, cwd=HERE, timeout=timeout).returncode


def audit_one(png, sid, tag):
    work = f"{D}/_pick_{sid}_{tag}"
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work, exist_ok=True)
    shutil.copy2(png, f"{work}/{sid}.png")

    ocr = f"{work}/audit.json"
    vl = f"{work}/vl.json"
    sc = f"{work}/sc.json"
    zm = f"{work}/zm.json"
    sem = f"{work}/sem.json"
    fin = f"{work}/final.json"

    code_l1 = f'''
import sys, os, json, glob
sys.path.insert(0, {HERE!r})
import tmp_text_rule_v5 as R
A = R.apply()
A.ROOT = {work!r}
A.CROP_DIR = {work!r} + "/_crops"
A.OUT_JSON = {ocr!r}
A.EXPECTED = A.load_expected()
A.init_readers(use_gpu=True)
results = []
for p in sorted(glob.glob(os.path.join(A.ROOT, "*.png"))):
    try:
        results.append(A.audit_one(p))
    except Exception as ex:
        results.append({{"file": os.path.basename(p), "verdict": "无文字",
                        "reason": str(ex), "ocr_strings": [], "crops": [],
                        "n_text_regions": 0, "similarity": 0.0, "expected": "",
                        "pseudo_strings": [], "real_strings": [], "text_regions": [],
                        "matched": False, "best_match": ""}})
json.dump({{"meta": {{"scenes_dir": A.ROOT}}, "summary": {{}}, "items": results}},
          open(A.OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("[L1]", [(r.get("verdict"), r.get("best_match")) for r in results], flush=True)
'''
    if sh(code_l1) != 0:
        return None

    for mod, out in (("tmp_vl_verify", vl), ("tmp_vl_signcheck", sc),
                     ("tmp_vl_zoom", zm)):
        code = f'''
import os, sys
sys.path.insert(0, {HERE!r})
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import {mod} as M
M.ROOT = {work!r}
M.CROP_DIR = {work!r} + "/_crops"
M.OUT = {out!r}
sys.argv = ["{mod}", {sid!r}]
M.main()
'''
        if sh(code) != 0:
            return None

    code_sem = f'''
import os, sys
sys.path.insert(0, {HERE!r})
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import transformers
_orig = transformers.PreTrainedModel.from_pretrained.__func__
def _patched(cls, *a, **k):
    if k.get("device_map") == "cuda:0":
        k["device_map"] = "cpu"
    return _orig(cls, *a, **k)
transformers.PreTrainedModel.from_pretrained = classmethod(_patched)
import tmp_vl_semantic as M
M.VL_JSON = {vl!r}
M.OUT = {sem!r}
M.main()
'''
    if sh(code_sem, cpu=True) != 0:
        return None

    code_fin = f'''
import sys
sys.path.insert(0, {HERE!r})
import tmp_text_rule_v5 as R
J = R.apply_adjudicate()
J.OCR_JSON = {ocr!r}
J.VL_JSON  = {vl!r}
J.SC_JSON  = {sc!r}
J.ZM_JSON  = {zm!r}
J.SEM_JSON = {sem!r}
J.OUT      = {fin!r}
J.main()
'''
    if sh(code_fin) != 0:
        return None

    d = json.load(open(fin, encoding="utf-8"))
    it = (d.get("items") or [{}])[0]
    return {"cand": png, "verdict": it.get("verdict"),
            "confidence": it.get("confidence"),
            "pseudo_evidence": it.get("pseudo_evidence"),
            "reason": it.get("reason"),
            "summary": d.get("summary")}


def main():
    sid = sys.argv[1]
    prefix = sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    cands = [f"{CAND}/{prefix}{i}.png" for i in range(n)
             if os.path.exists(f"{CAND}/{prefix}{i}.png")]
    print(f"=== {sid}: {len(cands)} 个候选 ===", flush=True)
    results = []
    for i, c in enumerate(cands):
        print(f"\n----- {os.path.basename(c)} -----", flush=True)
        r = audit_one(c, sid, f"t{i}")
        if r:
            results.append(r)
            print(f"  => {r['verdict']}  pseudo={r['pseudo_evidence']}", flush=True)

    out = f"{CAND}/{sid}_audit_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    print("\n=== 汇总 ===", flush=True)
    good = []
    for r in results:
        print(f"  {os.path.basename(r['cand']):32s} {str(r['verdict']):6s} "
              f"pseudo={r['pseudo_evidence']}", flush=True)
        if r["verdict"] == "真实文字":
            good.append(r)
    if good:
        print(f"\n>>> 合格: {good[0]['cand']}", flush=True)
    else:
        print("\n>>> 无合格候选", flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
