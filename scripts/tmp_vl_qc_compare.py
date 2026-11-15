#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双图直接对比质检（改进版）。

要点：
  * 让 VL 同时看两张图做逐项对照，而不是先描述再比较——
    通用描述词（椭圆脸/杏仁眼）无法区分不同人。
  * 明确告知：背面/侧后视图看不到脸时答"不可比"，不要答"否"。
  * 表情剧烈变化（愤怒/大笑）时，判据只看骨相与固定特征，不看表情。

输出三种结论：是 / 否 / 不可比
"""
import base64, json, os, re, sys, urllib.request

VL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen2.5-vl-72b"

COMPARE_PROMPT = """这两张照片里的人物，需要你判断是不是同一个人。

请按以下步骤在脑中对两张图做**逐项比对**（不要分别描述后再比，要直接对比）：

1) 先看第二张图能否看清面部。如果第二张是背面、侧后方、或脸被遮挡看不清五官，**直接回答"不可比"**。
2) 如果能看清，则逐项对比第一张和第二张：
   - 下颌线与下颌角的形状、宽窄
   - 眉形（走向、浓淡、眉峰位置、两眉间距）
   - 眼睛：眼型、眼距、双眼皮褶皱走向、大小
   - 鼻子：鼻梁高低与宽窄、鼻头形状、鼻翼宽度
   - 嘴唇：厚薄、唇形、嘴角走向
   - 脸型：颧骨宽窄、下巴尖圆
   - 特殊标记：痣、疤、酒窝、眼镜
3) 以上各项**全部吻合**才是同一人。只要有 2 项以上明显不同，就不是同一人。

注意：
- 表情不同（愤怒/大笑/哭泣）不代表换了人，只看面部结构
- 拍摄角度、镜头远近、光线、发丝飘动都不算差异
- 两个人都长成"椭圆脸杏仁眼"是常见巧合，必须找**具体**的差异点
  （比如鼻头形状、眼距、眉峰位置、痣的位置）

只回答三个词之一：是 / 否 / 不可比
"""


def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def ask(prompt, imgs, maxtok=20):
    content = [{"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64(p)}"}} for p in imgs]
    content.append({"type": "text", "text": prompt})
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}],
                       "max_tokens": maxtok, "temperature": 0.0}).encode()
    req = urllib.request.Request(VL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def verdict(a_path, b_path):
    t = ask(COMPARE_PROMPT, [a_path, b_path], 20).strip()
    if "不可比" in t:
        return "不可比"
    if re.search(r"^是|是$", t.strip()) or t.strip() == "是":
        return "是"
    if "否" in t:
        return "否"
    m = re.search(r"(是|否|不可比)", t)
    return m.group(1) if m else "?"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    args = ap.parse_args()
    print(verdict(args.a, args.b))
