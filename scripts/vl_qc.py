#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen2.5-VL 背景图质检：批量读图，输出结构化 JSONL。

模型常驻，单次加载处理整个目录。

用法:
  vl_qc.py --dir /ssd2/duanju/_probe_b100 --out /ssd2/duanju/_probe_b100/vl.jsonl
  vl_qc.py --dir ... --out ... --question "读出图中的所有文字"
"""
import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = '/ssd2/models/Qwen2.5-VL-7B-Instruct'

DEFAULT_Q = (
    "这是一张短剧用的背景图。请严格按 JSON 输出，不要任何多余文字："
    '{"scene": "一句话描述场景", '
    '"has_text": "有/无", '
    '"text_transcription": ["逐个列出画面中所有可见的文字（招牌/路牌/海报/书法/字样），原文照抄，读不出就写空字符串"], '
    '"text_locations": ["每处文字在画面中的位置，如 上部偏左"], '
    '"blank_sign_area": "描述画面中适合后期贴文字招牌的空白区域及其大致位置，没有就写无", '
    '"composition": "构图评价：主体位置/留白/是否适合竖屏短剧做背景", '
    '"problems": ["明显问题：伪文字、畸形结构、乱码、穿帮等，没有就空数组"]}'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--question', default=DEFAULT_Q)
    ap.add_argument('--question_file', default=None)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max_new_tokens', type=int, default=600)
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    if args.limit:
        files = files[:args.limit]
    question = args.question
    if args.question_file:
        with open(args.question_file, encoding='utf-8') as f:
            question = f.read().strip()
    print(f'[vl] {len(files)} images from {args.dir}', flush=True)

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa')
    model.eval()
    print(f'[vl] model loaded in {time.time()-t0:.0f}s', flush=True)

    with open(args.out, 'w', encoding='utf-8') as fout:
        for i, fn in enumerate(files):
            path = os.path.join(args.dir, fn)
            try:
                img = Image.open(path).convert('RGB')
            except Exception as ex:
                print(f'[vl] skip {fn}: {ex}', flush=True)
                continue
            msgs = [{'role': 'user', 'content': [
                {'type': 'image', 'image': img},
                {'type': 'text', 'text': question}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=[img], return_tensors='pt').to(model.device)
            t1 = time.time()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            ans = proc.batch_decode(out[:, inputs['input_ids'].shape[1]:],
                                    skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            rec = {'file': fn, 'path': path, 'size': list(img.size), 'answer': ans, 'secs': round(time.time() - t1, 1)}
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            fout.flush()
            print(f'[vl] {i+1}/{len(files)} {fn} ({rec["secs"]}s)', flush=True)

    print(f'[vl] done in {time.time()-t0:.0f}s -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
