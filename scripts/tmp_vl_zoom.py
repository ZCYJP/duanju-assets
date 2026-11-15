#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对候选图的文字区域做 4 倍高倍放大逐字符复核（Qwen2.5-VL-7B）。

目的：判定拉丁字母串是否为"把真单词画错"的伪文字。
输出 /ssd2/duanju/scenes_100_text_zoomcheck.json
"""
import base64
import io
import json
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = "/ssd2/models/Qwen2.5-VL-7B-Instruct"
ROOT = "/ssd2/duanju/scenes_100"
CROP_DIR = "/ssd2/duanju/bg_text_crops"
OUT = "/ssd2/duanju/scenes_100_text_zoomcheck.json"

PROMPT = (
    "这张图是场景背景招牌的 4 倍放大图。请非常仔细地做以下判断：\n"
    "1) 只报告你**确实能看清**的文字。看不清的一律写 [无法辨认]，"
    "绝对不要猜测、不要联想场景可能出现的字或店名。\n"
    "2) 对每个你能看清的拉丁字母串，判断是：\n"
    "   [REAL]   = 一个真实存在的英文/法文等单词，字母书写正确\n"
    "   [FAKE]   = 伪文字：某个真单词被画错了（字母被替换成形近字、多出杂字符、"
    "笔画扭曲、镜像），例如把 EXACT 画成 EXAACT\n"
    "3) 对每个你能看清的中文/日文/韩文串，先照实逐字抄写（不要纠正、不要脑补），"
    "再标注 [CJK]。\n"
    "严格每行一条输出：<原文> | [REAL]/[FAKE]/[CJK]/[无法辨认]\n不要解释。\n"
)


def b64(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    sids = sys.argv[1:]
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    out = []
    for sid in sids:
        cdir = os.path.join(CROP_DIR, sid)
        if not os.path.isdir(cdir):
            continue
        crops = sorted(f for f in os.listdir(cdir) if f.endswith(".png"))
        imgs = []
        for f in crops[:6]:
            im = Image.open(os.path.join(cdir, f)).convert("RGB")
            w, h = im.size
            k = max(1, min(4, 1600 // max(w, 1)))
            imgs.append(b64(im.resize((w * k, h * k), Image.LANCZOS)))
        if not imgs:
            continue
        content = [{"type": "image"} for _ in imgs]
        content.append({"type": "text", "text": PROMPT})
        msgs = [{"role": "user", "content": content}]
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[t], images=[imgs], padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            o = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        ans = proc.batch_decode([o[i][inputs.input_ids.shape[1]:] for i in range(len(o))],
                                 skip_special_tokens=True)[0]
        lines = [l.strip() for l in ans.splitlines() if l.strip()]
        out.append({"sid": sid, "crops": crops[:6], "answer": ans, "lines": lines})
        print(f"=== {sid} ({len(crops)} crops) ===")
        for l in lines:
            print("   ", l)
        print(flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()
