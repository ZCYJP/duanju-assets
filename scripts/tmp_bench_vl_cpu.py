#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测量 Qwen2.5-VL-7B 在 CPU 上做性别判定的耗时，判断能否替代 GPU 自检。"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = "/ssd2/models/Qwen2.5-VL-7B-Instruct"
PROMPT = ("看这张照片中的人物，只根据画面里实际呈现的相貌、体态和穿着判断："
          "这个人看起来是男性还是女性？\n"
          "第一行只输出一个词：男 或 女\n第二行用不超过25个字说明依据\n")

torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 32)
print(f"torch threads={torch.get_num_threads()}", flush=True)

t0 = time.time()
proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256*28*28, max_pixels=768*28*28)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32)
model.eval()
print(f"模型加载 {time.time()-t0:.1f}s", flush=True)

tests = [
    ("/ssd2/duanju/characters/cast/F36/F36_hero.png", "预期女（原图被判男）"),
    ("/ssd2/duanju/characters/cast/M10/M10_hero.png", "预期男"),
    ("/ssd2/duanju/characters/cast/F37/F37_hero.png", "预期女"),
]
for p, exp in tests:
    if not os.path.exists(p):
        print(f"skip {p}"); continue
    img = Image.open(p).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt")
    t1 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
    ans = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    print(f"{os.path.basename(p)} [{exp}] {time.time()-t1:.1f}s -> {ans.replace(chr(10),' | ')[:100]}", flush=True)
