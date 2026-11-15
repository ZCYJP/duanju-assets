#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""针对可疑背景图的逐招牌精细复核（Qwen2.5-VL-7B）。

对指定 sid 逐块裁剪放大，询问每块招牌是否含真实可读字符。
输出 /ssd2/duanju/scenes_100_text_signcheck.json
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
OUT = "/ssd2/duanju/scenes_100_text_signcheck.json"
CROP_DIR = "/ssd2/duanju/bg_text_crops"

PROMPT = (
    "这是一张短剧场景背景图的局部放大。请判断图中招牌/标牌上的文字：\n"
    "A) 存在一处或多处招牌文字；\n"
    "B) 请逐块列出你读到的文字原文（保持原语言，繁体照写繁体）；\n"
    "C) 对每一块，请标注该处文字是「真实字符」还是「伪文字（乱码/扭曲/镜像/形近字堆叠）」；\n"
    "D) 若该块没有文字，标注「无文字」。\n"
    "严格按如下每行一条格式输出，不要多余解释：\n"
    "文字 | 真实字符/伪文字/无文字\n"
)


def b64_img(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    sids = sys.argv[1:]
    if not sids:
        print("usage: tmp_vl_signcheck.py <sid>...")
        return
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    out = []
    for sid in sids:
        p = os.path.join(ROOT, f"{sid}.png")
        if not os.path.exists(p):
            continue
        im = Image.open(p).convert("RGB")
        W, H = im.size
        # 6 宫格 + 全图，每块 2x
        imgs = []
        for r in range(2):
            for c in range(3):
                box = (c * W // 3, r * H // 2, (c + 1) * W // 3, (r + 1) * H // 2)
                sub = im.crop(box).resize(((box[2] - box[0]) * 2, (box[3] - box[1]) * 2),
                                          Image.LANCZOS)
                imgs.append(b64_img(sub))
        imgs.append(b64_img(im))

        content = [{"type": "image"} for _ in imgs]
        content.append({"type": "text", "text": PROMPT})
        msgs = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[imgs], padding=True, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            o = model.generate(**inputs, max_new_tokens=220, do_sample=False)
        ans = proc.batch_decode(
            [o[i][inputs.input_ids.shape[1]:] for i in range(len(o))],
            skip_special_tokens=True)[0]
        lines = [l.strip() for l in ans.splitlines() if l.strip()]
        out.append({"sid": sid, "answer": ans, "lines": lines})
        print(f"=== {sid} ===")
        for l in lines:
            print("   ", l)
        print(flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()
