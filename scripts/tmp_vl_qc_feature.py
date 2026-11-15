#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""特征对照式质检：替代「是不是同一人」二选一，改为逐项特征核对。

二选一的问题：
  * 背面视图看不到脸，必然答「否」
  * 表情剧烈变化（angry/smile/sad）时五官比例改变，容易答「否」
  * 跨性别的相似脸型容易答「是」（原版误报的根源）

特征对照的做法：让 VL 分别描述两张图的结构性特征，再逐项比较，
只有「骨相结构 + 关键辨识特征」都对上才判定为同一人。
表情、发型随角度的常规变化、拍摄背景都不计入判据。
"""
import base64, json, os, re, sys, urllib.request

VL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen2.5-vl-72b"

FEATURE_PROMPT = """请只描述这张图片中人物面部的**结构性特征**，用一行文字，不要评价、不要推测性格。

按这个格式输出：
脸型:__ ; 额头:__ ; 眉形:__ ; 眼型:__ ; 眼距:__ ; 鼻型:__ ; 嘴型:__ ; 下颌线:__ ; 脸型不对称特征:__ ; 肤色:__

注意：
- 只写你确实看到的特征，看不清的写"不清"
- 忽略表情、发型、年龄感、服装、背景、灯光
- 脸型不对称特征请写具体位置（如"左颧骨略高"），没有就写"无明显"
"""


def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def ask(prompt, imgs, maxtok=300):
    content = [{"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64(p)}"}} for p in imgs]
    content.append({"type": "text", "text": prompt})
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}],
                       "max_tokens": maxtok, "temperature": 0.0}).encode()
    req = urllib.request.Request(VL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def parse_feat(t):
    d = {}
    for part in re.split(r'[;；]', t):
        if ':' in part or '：' in part:
            k, v = re.split(r'[:：]', part, 1)
            d[k.strip()] = v.strip()
    return d


# 关键特征项：这些是判定同一人的核心依据
CORE = ["脸型", "眼型", "眼距", "鼻型", "嘴型", "下颌线"]
# 允许一定差异的项（受角度/表情影响大）
SOFT = ["额头", "眉形", "脸型不对称特征", "肤色"]


def compare(fa, fb):
    """逐项比较两份特征描述，返回 (是否同一人, 明细)。"""
    same_core, diff_core, soft_ok = 0, [], 0
    for k in CORE:
        va, vb = fa.get(k, ""), fb.get(k, "")
        if not va or not vb or "不清" in va or "不清" in vb:
            continue
        if va == vb:
            same_core += 1
        elif any(x in va for x in vb) or any(x in vb for x in va):
            same_core += 0.5
        else:
            diff_core.append((k, va, vb))
    for k in SOFT:
        va, vb = fa.get(k, ""), fb.get(k, "")
        if va and vb and va == vb:
            soft_ok += 1
    total = len([k for k in CORE if fa.get(k) and fb.get(k) and "不清" not in fa.get(k, "")])
    ratio = (same_core / total) if total else 0.0
    detail = {"core_same": same_core, "core_total": total, "ratio": round(ratio, 2),
              "soft_match": soft_ok, "diff": diff_core}
    # 判定：核心特征 75% 以上一致，且软特征至少 1 项一致
    is_same = total >= 3 and ratio >= 0.75 and soft_ok >= 1
    return is_same, detail


def verify_pair(a_path, b_path, label=""):
    fa = parse_feat(ask(FEATURE_PROMPT, [a_path], 220))
    fb = parse_feat(ask(FEATURE_PROMPT, [b_path], 220))
    same, detail = compare(fa, fb)
    detail["feat_a"] = fa
    detail["feat_b"] = fb
    detail["label"] = label
    return same, detail


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    args = ap.parse_args()
    same, d = verify_pair(args.a, args.b)
    print(json.dumps(d, ensure_ascii=False, indent=2))
    print("SAME" if same else "DIFF")
