#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VL 复核：对 OCR 结果存疑的背景图，用 Qwen2.5-VL-7B 直接读招牌文字。

输出 /ssd2/duanju/scenes_100_text_vl_verify.json
用法: python tmp_vl_verify.py <sid1> <sid2> ...
      python tmp_vl_verify.py --all     # 复核全部 100 张
"""
import base64
import json
import os
import re
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = "/ssd2/models/Qwen2.5-VL-7B-Instruct"
ROOT = "/ssd2/duanju/scenes_100"
OUT = "/ssd2/duanju/scenes_100_text_vl_verify.json"
SRC_SCENES = "/ssd2/duanju/scripts/gen_scenes_100.py"

PROMPT = (
    "请仔细阅读这张图里所有招牌/标牌上的文字，逐条列出你看到的文字内容。\n"
    "要求：\n"
    "1) 只输出你真正能辨认的文字本身，一行一条，不要解释、不要加引号；\n"
    "2) 如果某处文字是乱码、伪文字、扭曲变形或镜像的字符，请照实写出来并在行末标注 [伪文字]；\n"
    "3) 如果完全没有文字，输出 [无文字]。\n"
)


def load_expected():
    src = open(SRC_SCENES, encoding="utf-8").read()
    m = re.search(r"SCENES = \[(.*?)\n\]", src, re.S)
    ns = {}
    exec("SCENES = [" + m.group(1) + "\n]", ns)
    return {r[0]: {"expected": r[3] or "", "lang": r[4], "cn": r[1]} for r in ns["SCENES"]}


def norm(s):
    return re.sub(r"\s+", "", s or "")


def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    expected = load_expected()
    args = sys.argv[1:]
    if args and args[0] == "--all":
        sids = sorted(os.path.splitext(f)[0] for f in os.listdir(ROOT)
                      if f.endswith(".png"))
    else:
        sids = args
    if not sids:
        print("no sids")
        return

    print(f"[vl] loading {MODEL}", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    print("[vl] model ready", flush=True)

    results = []
    for i, sid in enumerate(sids):
        p = os.path.join(ROOT, f"{sid}.png")
        if not os.path.exists(p):
            print(f"  ! missing {sid}", flush=True)
            continue
        try:
            im = Image.open(p).convert("RGB")
            # 整图 + 局部放大（招牌通常只占画面一小块）
            full_b64 = img_b64(p)
            crops = []
            W, H = im.size
            # 上半/中/下 三条横带，放大 2 倍
            for k in range(3):
                y0 = int(H * k / 3)
                y1 = int(H * (k + 1) / 3)
                sub = im.crop((0, y0, W, y1)).resize((W * 2, (y1 - y0) * 2), Image.LANCZOS)
                import io
                buf = io.BytesIO()
                sub.save(buf, format="PNG")
                crops.append(base64.b64encode(buf.getvalue()).decode())

            content = [{"type": "image"} for _ in range(1 + len(crops))]
            content.append({"type": "text", "text": PROMPT})
            msgs = [{"role": "user", "content": content}]
            texts = [proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)]
            img_lists = [[full_b64] + crops]

            inputs = proc(text=texts, images=img_lists, padding=True, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
            trimmed = [out[i][inputs.input_ids.shape[1]:] for i in range(len(out))]
            answer = proc.batch_decode(trimmed, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)[0]
        except Exception as e:
            import traceback
            traceback.print_exc()
            answer = f"[ERROR] {e}"

        lines = [l.strip() for l in answer.splitlines() if l.strip()]
        exp = expected.get(sid, {}).get("expected", "")
        rec = {
            "sid": sid, "expected": exp,
            "lang": expected.get(sid, {}).get("lang", ""),
            "vl_answer": answer, "vl_lines": lines,
            "has_pseudo_marker": any("[伪文字]" in l for l in lines),
            "says_no_text": any("[无文字]" in l for l in lines),
        }
        results.append(rec)
        print(f"  [{i+1}/{len(sids)}] {sid} exp={exp!r} -> {lines}", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[done] -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
