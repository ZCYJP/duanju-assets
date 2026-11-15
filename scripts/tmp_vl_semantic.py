#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""语义级伪文字判定：对 VL 读到的拉丁字母串，判断是否为"扭曲变形的伪单词"。

背景：AI 生成图常把 "EXACT" 画成 "EXAACT"、把 "PROFESSIONAL" 画成
"FABLISATESSIC"。这些串的每个字符都是合法 ASCII，按纯字符集规则会误判为
"真实字符"，但它们是形近字堆叠/镜像扭曲的伪文字。

本脚本用 VL 对候选串做二次追问：问"这个词是否是一个真实存在的单词，
还是把单词画错后的乱码"，并给出独立判定。

用法: python tmp_vl_semantic.py
输入: /ssd2/duanju/scenes_100_text_vl_verify.json (取 VL 读到的串)
输出: /ssd2/duanju/scenes_100_text_semantic.json
"""
import json
import os
import re
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = "/ssd2/models/Qwen2.5-VL-7B-Instruct"
ROOT = "/ssd2/duanju/scenes_100"
VL_JSON = "/ssd2/duanju/scenes_100_text_vl_verify.json"
OUT = "/ssd2/duanju/scenes_100_text_semantic.json"

PROMPT = (
    "下面这些字符串是从一张 AI 生成的场景背景图的招牌上读出来的。\n"
    "请逐条判断每一项属于以下哪一类：\n"
    "  [OK]      = 一个真实存在的单词/词组/真实文字，读法自然\n"
    "  [PSEUDO]  = 伪文字：把真实单词画错后的乱码（形近字替换、字母镜像、"
    "笔画扭曲、多余杂字符），例如 EXACT 被画成 EXAACT、PROFESSIONAL 被画成 "
    "FABLISATESSIC 这类\n"
    "  [CJK]     = 真实的中文/日文/韩文文字\n"
    "严格按每行一条输出：<原字符串> | [OK]/[PSEUDO]/[CJK]\n"
    "不要输出解释。\n\n"
)


def norm(s):
    return re.sub(r"\s+", "", s or "")


def has_cjk(s):
    return any(0x4E00 <= ord(c) <= 0x9FFF or 0x3040 <= ord(c) <= 0x30FF
               or 0xAC00 <= ord(c) <= 0xD7AF for c in s)


def main():
    vl = json.load(open(VL_JSON, encoding="utf-8"))
    # 收集所有候选拉丁串（>=3 字符，含空格分词）
    cands = []
    seen = set()
    for r in vl:
        for line in r.get("vl_lines", []):
            s = line.replace("[伪文字]", "").strip()
            s = re.sub(r"\|.*$", "", s).strip()
            if not s or s.startswith("[") or has_cjk(s):
                continue
            if len(norm(s)) < 3:
                continue
            if s in seen:
                continue
            seen.add(s)
            cands.append(s)
    print(f"[semantic] 待判候选 {len(cands)} 条", flush=True)
    if not cands:
        print("[done] 无需判定")
        return

    proc = AutoProcessor.from_pretrained(MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    results = {}
    # 分批，每批 30 条
    for i in range(0, len(cands), 30):
        batch = cands[i:i + 30]
        text = PROMPT + "\n".join(f"{j+1}. {c}" for j, c in enumerate(batch))
        msgs = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        t = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[t], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            o = model.generate(**inputs, max_new_tokens=700, do_sample=False)
        ans = proc.batch_decode([o[0][inputs.input_ids.shape[1]:]],
                                 skip_special_tokens=True)[0]
        for line in ans.splitlines():
            line = line.strip()
            if "|" not in line:
                continue
            left, right = line.split("|", 1)
            left = left.strip()
            right = right.strip().upper()
            key = re.sub(r"^\d+[.)]\s*", "", left).strip()
            tag = "PSEUDO" if "PSEUDO" in right else ("CJK" if "CJK" in right else
                                                       ("OK" if "OK" in right else "?"))
            if key:
                results[key] = tag
        print(f"  batch {i//30+1}/{(len(cands)+29)//30} done", flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"candidates": cands, "verdicts": results}, f,
                  ensure_ascii=False, indent=2)

    pseudo = sorted(k for k, v in results.items() if v == "PSEUDO")
    ok = sorted(k for k, v in results.items() if v == "OK")
    unk = sorted(k for k, v in results.items() if v == "?")
    print(f"\n[semantic] PSEUDO({len(pseudo)}): {pseudo}")
    print(f"[semantic] OK({len(ok)}): {ok}")
    if unk:
        print(f"[semantic] 未判定({len(unk)}): {unk}")
    print(f"[done] -> {OUT}")


if __name__ == "__main__":
    main()
