#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐裁剪块单标签判定（Qwen2.5-VL-7B）。

对每张背景图的每个文字裁剪块**单独**发起一次调用，要求模型**只选一个**标签，
避免多图多标签时模型回显整个标签表造成的误判。

标签:
  A = 真实字符（文字写得正确、可正常阅读）
  B = 伪文字（形近字替换 / 笔画扭曲 / 镜像 / 杂字符；即"把真字画错了"）
  C = 无法辨认（分辨率不足或字迹严重变形，无法判断）

输出 /ssd2/duanju/scenes_100_text_zoom2.json
"""
import base64
import io
import json
import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = "/ssd2/models/Qwen2.5-VL-7B-Instruct"
CROP_DIR = "/ssd2/duanju/bg_text_crops"
OUT = "/ssd2/duanju/scenes_100_text_zoom2.json"

PROMPT = (
    "这是一张场景背景图中某个招牌/标牌区域的高倍放大图。\n"
    "请只回答一个字母，不要写任何其它内容：\n"
    "A = 图中文字是真实字符，书写正确、能正常读出来\n"
    "B = 图中文字是伪文字：某个真单词被画错了（字母/汉字被替换成形近字、"
    "笔画扭曲、镜像翻转、混入杂字符），例如 EXACT 被画成 EXAACT\n"
    "C = 图中文字完全无法辨认（太模糊或变形到无法判断是字还是图案）\n"
    "只输出 A、B 或 C 中的一个字母："
)


def b64(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    sids = sys.argv[1:]
    if not sids:
        print("usage: tmp_vl_zoom2.py <sid>...")
        return
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    out = []
    total = 0
    for sid in sids:
        cdir = os.path.join(CROP_DIR, sid)
        if not os.path.isdir(cdir):
            continue
        crops = sorted(f for f in os.listdir(cdir) if f.endswith(".png"))
        recs = []
        for f in crops:
            im = Image.open(os.path.join(cdir, f)).convert("RGB")
            w, h = im.size
            k = max(1, min(4, 1400 // max(w, 1)))
            im2 = im.resize((w * k, h * k), Image.LANCZOS)
            msgs = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": PROMPT}]}]
            t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[t], images=[b64(im2)], return_tensors="pt").to(model.device)
            with torch.inference_mode():
                o = model.generate(**inputs, max_new_tokens=5, do_sample=False)
            ans = proc.batch_decode([o[0][inputs.input_ids.shape[1]:]],
                                     skip_special_tokens=True)[0].strip()
            label = "C"
            for ch in ans.upper():
                if ch in "ABC":
                    label = ch
                    break
            recs.append({"crop": f, "label": label, "raw": ans})
            total += 1
            time.sleep(0.02)
        out.append({"sid": sid, "crops": recs})
        nB = sum(1 for r in recs if r["label"] == "B")
        flag = "  <<< FAKE" if nB else ""
        print(f"  {sid:<24} blocks={len(recs)} A={sum(1 for r in recs if r['label']=='A')} "
              f"B={nB} C={sum(1 for r in recs if r['label']=='C')}{flag}", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[done] {total} crops -> {OUT}")


if __name__ == "__main__":
    main()
