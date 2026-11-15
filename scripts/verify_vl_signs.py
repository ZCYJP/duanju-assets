#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最终汉字验收：对放大后的牌面逐张用 VL 回读，与预期文案比对。

这是"汉字是否真实"的最终判据 —— 读得出且与预期一致 = 真字；
读不出或对不上 = 该图需重做。

用法: verify_vl_signs.py --zoom_dir /ssd2/duanju/_sign_zoom \
                       --scenes /ssd2/duanju/scripts/scenes_b100.json \
                       --out /ssd2/duanju/bg100_vl_signs.json
"""
import argparse
import difflib
import json
import os
import re

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = '/ssd2/models/Qwen2.5-VL-7B-Instruct'
QUESTION = ('这是一张招牌牌匾的放大特写。请只回答这行字写的是什么，直接输出文字本身，'
            '不要解释、不要加引号。如果字迹模糊或无法辨认，就输出"无法辨认"。')


def norm(s):
    s = re.sub(r'[\s"\'\'"「」『』]', '', s or '')
    return s.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zoom_dir', required=True)
    ap.add_argument('--scenes', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--suffix', default='')
    args = ap.parse_args()

    with open(args.scenes, encoding='utf-8') as f:
        scenes = json.load(f)['scenes']
    todo = [s for s in scenes if s.get('sign')]

    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28, max_pixels=1600 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa')
    model.eval()

    results = []
    for i, sc in enumerate(todo):
        sid = sc['sid']
        exp = sc['sign']['text']
        path = os.path.join(args.zoom_dir, f'{sid}{args.suffix}.png')
        if not os.path.exists(path):
            results.append({'sid': sid, 'expected': exp, 'verdict': 'MISSING'})
            continue
        img = Image.open(path).convert('RGB')
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img},
                                             {'type': 'text', 'text': QUESTION}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors='pt').to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        ans = proc.batch_decode(out[:, inputs['input_ids'].shape[1]:],
                                skip_special_tokens=True)[0].strip()
        got = norm(ans)
        e = norm(exp)
        ratio = difflib.SequenceMatcher(None, got, e).ratio() if got and e else 0.0
        if got == e:
            verdict = 'PASS'
        elif e and (e in got or got in e) and len(e) >= 2:
            verdict = 'PASS_PARTIAL'
        elif ratio >= 0.6:
            verdict = 'NEAR'
        else:
            verdict = 'FAIL'
        results.append({'sid': sid, 'expected': exp, 'got': got, 'raw': ans,
                        'ratio': round(ratio, 3), 'verdict': verdict,
                        'style': sc['sign'].get('style')})
        print(f"{verdict:12} {sid:26} exp={exp!r} got={got!r} r={ratio:.2f}", flush=True)

    n_pass = sum(1 for r in results if r['verdict'] in ('PASS', 'PASS_PARTIAL'))
    summary = {'total': len(results), 'pass': n_pass, 'fail': len(results) - n_pass,
               'failed_sids': [r['sid'] for r in results if r['verdict'] not in ('PASS', 'PASS_PARTIAL')]}
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'items': results}, f, ensure_ascii=False, indent=1)
    print('=== summary ===', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
