#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对被判为 FAKE 的裁剪块做独立复核（不同提示词 + 反向问法 + 多数表决）。

目的：排除 VL 单次判定的噪声。只保留多次独立判定一致为伪文字的块。
输出 /ssd2/duanju/scenes_100_text_zoom3.json
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
OUT = "/ssd2/duanju/scenes_100_text_zoom3.json"

PROMPTS = [
    # 问法 1：直接二选一
    ("图中文字能否正常、正确地读出来？只能回答一个字母。\n"
     "A = 能正常读出来，写法正确\n"
     "B = 读出来像是把某个真单词/汉字画错了（形近字、笔画扭曲、镜像、乱字符）\n"
     "只回答 A 或 B："),
    # 问法 2：反向问法
    ("仔细看这张招牌放大图。图中这些字是不是 AI 生成时画错的假字（"
     "看起来像字但其实不是正确写法）？只能回答一个字母。\n"
     "A = 是画错的假字\n"
     "B = 不是，是写得正确的真字\n"
     "只回答 A 或 B："),
    # 问法 3：要求先描述再判断
    ("请做两件事：\n"
     "第一步：用一句话描述图中文字的书写特征（笔画是否扭曲、是否有镜像、"
     "是否有形近字替换、是否有多余杂字符）。\n"
     "第二步：最后一行只输出结论字母——A 表示真实字符，B 表示伪文字。"),
]


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
        recs = []
        for f in crops:
            im = Image.open(os.path.join(cdir, f)).convert("RGB")
            w, h = im.size
            k = max(1, min(4, 1400 // max(w, 1)))
            im2 = im.resize((w * k, h * k), Image.LANCZOS)
            votes = []
            raw = []
            for pi, prompt in enumerate(PROMPTS):
                msgs = [{"role": "user", "content": [
                    {"type": "image"}, {"type": "text", "text": prompt}]}]
                t = proc.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
                inputs = proc(text=[t], images=[b64(im2)],
                              return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    o = model.generate(**inputs, max_new_tokens=90, do_sample=False)
                ans = proc.batch_decode([o[0][inputs.input_ids.shape[1]:]],
                                         skip_special_tokens=True)[0].strip()
                raw.append(ans)
                # 问法1/2: A=真实,B=伪; 问法3: A=伪,B=真实
                lab = None
                for ch in ans.upper():
                    if ch in "AB":
                        lab = ch
                        break
                if lab:
                    fake = (lab == "B") if pi < 2 else (lab == "A")
                    votes.append("FAKE" if fake else "REAL")
                time.sleep(0.02)
            n_fake = votes.count("FAKE")
            n_real = votes.count("REAL")
            verdict = ("FAKE" if n_fake > n_real else
                       ("REAL" if n_real > n_fake else "AMBIG"))
            recs.append({"crop": f, "votes": votes, "verdict": verdict,
                         "n_fake": n_fake, "n_real": n_real, "raw": raw})
            if verdict == "FAKE":
                print(f"    {sid}/{f}: FAKE votes={votes}", flush=True)
        nF = sum(1 for r in recs if r["verdict"] == "FAKE")
        print(f"  {sid:<24} blocks={len(recs)} FAKE={nF}", flush=True)
        out.append({"sid": sid, "crops": recs})

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()
