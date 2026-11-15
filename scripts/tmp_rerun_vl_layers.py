#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 scenes_100_regen/ 的 9 张重生成背景，重跑完整四层文字核验（L2/L3/L4 + 裁决）。

与原流程口径完全一致：
  L1 OCR 字符集硬规则      -> 已由 tmp_audit_regen.py 产出 scenes_100_regen_text_audit.json
  L2 VL 整图转写            -> 本脚本产出 scenes_100_regen_text_vl_verify.json
  L3 VL 逐招牌真实/伪标注   -> 本脚本产出 scenes_100_regen_text_signcheck.json
  L4 VL 放大逐字符 + 语义    -> 本脚本产出 scenes_100_regen_text_zoomcheck.json / _semantic.json
  最终裁决                  -> 本脚本产出 scenes_100_regen_text_final.json

做法：把原脚本里的模块级常量 ROOT / OUT / CROP_DIR 覆盖为新路径后复用其 main()，
从而保证 prompt、判据、输出结构与原始流程字节级一致。
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = "/ssd2/duanju"
NEW_ROOT = f"{D}/scenes_100_regen"
NEW_CROPS = f"{NEW_ROOT}/_crops"
SIDS = ["bg_auto_shop", "bg_delivery_street", "bg_driving_office",
        "bg_grain_depot", "bg_home_services", "bg_kindergarten_paint",
        "bg_law_office", "bg_pet_shop", "bg_tofu_stall"]

PY = "/ssd2/envs/qwentts/bin/python"


def run_layer(mod, out_path, extra_env=None, sids=None, vl_json=None):
    """以 patch 常量的方式运行一个 VL 层脚本。"""
    code = f'''
import os, sys, json
sys.path.insert(0, {HERE!r})
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import {mod} as M
M.ROOT = {NEW_ROOT!r}
M.CROP_DIR = {NEW_CROPS!r}
M.OUT = {out_path!r}
if hasattr(M, "VL_JSON"):
    M.VL_JSON = {vl_json or out_path!r}
sys.argv = ["{mod}"] + {sids!r}
M.main()
'''
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    if extra_env:
        env.update(extra_env)
    print(f"--- layer {mod} -> {out_path} ---", flush=True)
    r = subprocess.run([PY, "-c", code], env=env, cwd=HERE)
    if r.returncode != 0:
        print(f"[layer {mod}] FAILED rc={r.returncode}", flush=True)
        return False
    return True


def main():
    os.makedirs(NEW_CROPS, exist_ok=True)

    # L2: VL 整图转写
    if not run_layer("tmp_vl_verify", f"{D}/scenes_100_regen_text_vl_verify.json",
                     sids=SIDS):
        return 1

    # L3: VL 逐招牌标注（需要 crops，来自 OCR 审计）
    if not run_layer("tmp_vl_signcheck", f"{D}/scenes_100_regen_text_signcheck.json",
                     sids=SIDS):
        return 1

    # L4a: VL 放大逐字符
    if not run_layer("tmp_vl_zoom", f"{D}/scenes_100_regen_text_zoomcheck.json",
                     sids=SIDS):
        return 1

    # L4b: 拉丁串语义判定（输入是 L2 的 vl_verify，不是 zoomcheck）
    if not run_layer("tmp_vl_semantic", f"{D}/scenes_100_regen_text_semantic.json",
                     vl_json=f"{D}/scenes_100_regen_text_vl_verify.json"):
        return 1

    # 最终裁决：复用 tmp_bg_text_adjudicate.py 的全部判据
    print("--- adjudicate ---", flush=True)
    code = f'''
import os, sys
sys.path.insert(0, {HERE!r})
import tmp_bg_text_adjudicate as J
J.OCR_JSON = "{D}/scenes_100_regen_text_audit.json"
J.VL_JSON  = "{D}/scenes_100_regen_text_vl_verify.json"
J.SC_JSON  = "{D}/scenes_100_regen_text_signcheck.json"
J.ZM_JSON  = "{D}/scenes_100_regen_text_zoomcheck.json"
J.SEM_JSON = "{D}/scenes_100_regen_text_semantic.json"
J.OUT      = "{D}/scenes_100_regen_text_final.json"
J.main()
'''
    r = subprocess.run([PY, "-c", code], cwd=HERE)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
