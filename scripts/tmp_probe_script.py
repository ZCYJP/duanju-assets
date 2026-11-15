#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：判断 VL 转录输出的是"图片上的真实字形"还是"被评委自动转成简体"。

方法：对同一张招牌同时问三个问题
  Q1 直接转录
  Q2 招牌是繁体还是简体
  Q3 强制二选一（针对易混字）
若 Q1 输出简体而 Q2 答"繁体"，说明图片是繁体、评委只是转录时规范化了。
"""
import base64, json, sys, urllib.request

VL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen2.5-vl-72b"


def img_b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def ask(prompt, img, maxtok=60):
    content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64(img)}"}}]
    content.append({"type": "text", "text": prompt})
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}],
                       "max_tokens": maxtok, "temperature": 0.0}).encode()
    req = urllib.request.Request(VL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


for sid, pairs in [
    ("bg_flower_stall", [("鲜花", "鮮花"), ("鲜", "鮮")]),
    ("bg_tailor_shop", [("裁缝店", "裁縫店")]),
    ("bg_tea_prep", [("調餐室", "調製室"), ("調制室", "調製室")]),
]:
    p = f"/ssd2/duanju/scenes_100/{sid}.png"
    print(f"===== {sid} =====")
    print(" Q1 转录:", ask("请读出图中招牌上的文字，只输出文字本身。", p))
    print(" Q2 繁简:", ask("图中招牌上的文字使用的是繁体字还是简体字？只回答'繁体'或'简体'。", p, 16))
    for a, b in pairs:
        q = f"图中招牌上是否包含字形「{a}」？还是「{b}」？只回答 {a} 或 {b} 之一。"
        print(f" Q3 {a}/{b}:", ask(q, p, 16))
