#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 CPU 上跑 tmp_vl_semantic（VL 拉丁串语义判定）。

背景：脚本硬编码 device_map="cuda:0"，而 GPU 0 被 ComfyUI 渲染队列占满（78.7G/81.5G），
导致 torch.OutOfMemoryError。该层每次只判 3~30 条候选串，192 核 CPU 足够快。

做法：monkeypatch transformers 的 from_pretrained，把 device_map 改为 "cpu"，
并强制 CUDA 不可见，避免任何 CUDA 分配。

用法: python tmp_vl_semantic_cpu.py <vl_json> <out_json>
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # 彻底屏蔽 CUDA
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

HERE = "/ssd2/duanju/scripts"
sys.path.insert(0, HERE)

VL_JSON = sys.argv[1] if len(sys.argv) > 1 else \
    "/ssd2/duanju/scenes_100_regen_text_vl_verify_v5.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    "/ssd2/duanju/scenes_100_regen_text_semantic_v5.json"

# patch transformers：device_map="cuda:0" -> "cpu"
import transformers
_orig = transformers.PreTrainedModel.from_pretrained.__func__


def _patched(cls, *args, **kwargs):
    if kwargs.get("device_map") == "cuda:0":
        kwargs["device_map"] = "cpu"
    return _orig(cls, *args, **kwargs)


transformers.PreTrainedModel.from_pretrained = classmethod(_patched)

import tmp_vl_semantic as M
M.VL_JSON = VL_JSON
M.OUT = OUT
M.main()
print(f"[semantic-cpu done] -> {OUT}", flush=True)
