#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""继续跑 100 张原始背景的 L2/L3/L4 + 裁决（v5 规则）。

L1 已由 tmp_full_reaudit_v5.py orig 完成：
  真实文字 99 / 伪文字 1 (bg_fresh_aisle) / 无文字 0

L2 失败原因：tmp_vl_verify.py 的 --all 模式用 os.listdir(ROOT) 扫目录，
  而 scenes_100/ 下还有 text_report_*.json 等非图片文件，
  splitext 后变成乱名 sid 传入。这里改用显式的 100 个 sid 列表。

用法: python tmp_reaudit_orig_v5_rest.py
"""
import json
import os
import subprocess
import sys

HERE = "/ssd2/duanju/scripts"
D = "/ssd2/duanju"
PY = "/ssd2/envs/qwentts/bin/python"
ROOT = f"{D}/scenes_100"
CROPS = f"{D}/bg_text_crops_v5"

SIDS = sorted(os.path.splitext(f)[0] for f in os.listdir(ROOT) if f.endswith(".png"))
assert len(SIDS) == 100, f"期望 100 个 sid，实得 {len(SIDS)}"


def sh(code, timeout=3000, cpu=False):
    e = dict(os.environ)
    if cpu:
        e["CUDA_VISIBLE_DEVICES"] = ""
    else:
        e["CUDA_VISIBLE_DEVICES"] = "1"
    r = subprocess.run([PY, "-c", code], env=e, cwd=HERE, timeout=timeout)
    return r.returncode


def layer(mod, out, cpu=False):
    print(f"\n=== {mod} -> {out} ===", flush=True)
    code = f'''
import os, sys
sys.path.insert(0, {HERE!r})
os.environ["CUDA_VISIBLE_DEVICES"] = {"" if cpu else '"1"'}
import {mod} as M
M.ROOT = {ROOT!r}
M.CROP_DIR = {CROPS!r}
M.OUT = {out!r}
if hasattr(M, "VL_JSON"):
    M.VL_JSON = {out!r}
sys.argv = ["{mod}"] + {SIDS!r}
M.main()
'''
    return sh(code, cpu=cpu)


def main():
    vl = f"{D}/scenes_100_text_vl_verify_v5.json"
    sc = f"{D}/scenes_100_text_signcheck_v5.json"
    zm = f"{D}/scenes_100_text_zoomcheck_v5.json"
    sem = f"{D}/scenes_100_text_semantic_v5.json"
    fin = f"{D}/scenes_100_text_final_v5.json"
    ocr = f"{D}/scenes_100_text_audit_v5.json"

    if layer("tmp_vl_verify", vl) != 0:
        print("[L2 FAILED]", flush=True)
        return 1
    if layer("tmp_vl_signcheck", sc) != 0:
        print("[L3 FAILED]", flush=True)
        return 1
    if layer("tmp_vl_zoom", zm) != 0:
        print("[L4a FAILED]", flush=True)
        return 1

    # L4b 语义层：CPU 版（GPU 被 ComfyUI 队列占满）
    print("\n=== tmp_vl_semantic (CPU) ===", flush=True)
    code = f'''
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
    if sh(code, cpu=True) != 0:
        print("[L4b FAILED]", flush=True)
        return 1

    print("\n=== adjudicate ===", flush=True)
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
J.OUT      = {fin!r}
J.main()
'''
    if sh(code) != 0:
        print("[adjudicate FAILED]", flush=True)
        return 1

    d = json.load(open(fin, encoding="utf-8"))
    print("\n=== FINAL ===", flush=True)
    print(json.dumps(d.get("summary", {}), ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
