#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""存疑项仲裁：InsightFace 嵌入相似度 + 第三票 VL 追问。

双盲交叉出现 ['是','否'] 时，说明评委存在顺序偏差（先看到哪张就倾向哪张），
不能直接判为"换人"。这里引入两个独立证据：
  1) InsightFace  buffalo_l 提取人脸嵌入，算与 hero 的余弦相似度（定量）
  2) 第三次 VL 询问，换一种问法并把两张图的顺序随机化（定性）

用法：
  python3 tmp_qc_arbitrate.py --qc /ssd2/duanju/seasons/qc_shard_0.json [...]
  python3 tmp_qc_arbitrate.py --qc-glob "/ssd2/duanju/seasons/qc_shard_*.json"
输出：仲裁后的 JSON 写到同名 .arbitrated.json
"""
import argparse
import base64
import glob
import json
import os
import re
import sys
import time
import urllib.request
import warnings

import cv2
import numpy as np
import insightface

warnings.filterwarnings('ignore')

VL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen2.5-vl-72b"
SEASON_ROOT = "/ssd2/duanju/seasons"
CAST_ROOT = "/ssd2/duanju/characters/cast"

_app = None


def get_app():
    global _app
    if _app is None:
        _app = insightface.app.FaceAnalysis(name="buffalo_l",
                                            providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=-1, det_size=(640, 640))
    return _app


def embed(path):
    app = get_app()
    im = cv2.imread(path)
    if im is None:
        return None
    fs = [x for x in app.get(im) if x.det_score > 0.5]
    if not fs:
        p = im.shape[0] // 4
        fs = [x for x in app.get(cv2.copyMakeBorder(im, p, p, p, p,
                                                   cv2.BORDER_CONSTANT, value=0))
              if x.det_score > 0.5]
    if not fs:
        return None
    f = max(fs, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return f.normed_embedding


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


# 换一种问法：不再问"是不是同一个人"，而是问"哪张图的五官更标准、更像
# 一张真实的同一人照片"，规避 yes/no 的顺序锚定。
PROMPT3 = """这两张是同一个人的两张照片。第一张和第二张中，只有一张里的脸
更可能是这个人的"标准脸"，另一张可能被生成模型替换成了另一个人的脸。

请重点看：眼距、眼型、鼻头形状、嘴唇轮廓、下颌线、眉峰位置。

请只回答一个词：第1张 / 第2张
（判断哪一张更像是真实的、未经替换的脸）
"""


def third_vote(a, b):
    t = ask(PROMPT3, [a, b], 16).strip()
    if "第1" in t or "1" == t:
        return "1"
    if "第2" in t:
        return "2"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", nargs="*", default=[])
    ap.add_argument("--qc-glob", default="")
    ap.add_argument("--sim-thresh", type=float, default=0.45,
                    help="人脸嵌入余弦相似度阈值；超过判为同一人")
    args = ap.parse_args()

    files = list(args.qc)
    if args.qc_glob:
        files += sorted(glob.glob(args.qc_glob))
    files = sorted(set(files))
    if not files:
        print("no qc files")
        return

    suspects = []
    seen = set()
    for f in files:
        if not os.path.exists(f):
            continue
        for r in json.load(open(f, encoding="utf-8")):
            if r.get("verdict") in ("否", "存疑", "?"):
                key = (r["cid"], r["variant"])
                if key in seen:
                    continue
                seen.add(key)
                suspects.append((f, r))

    print(f"[data] {len(suspects)} suspects from {len(files)} files", flush=True)
    out = []
    t0 = time.time()
    for n, (srcf, r) in enumerate(suspects):
        cid = r["cid"]
        variant = r["variant"]
        # variant 形如 F01_mangzhong，去掉前缀得到 key
        key = variant.split("_", 1)[1] if variant.startswith(cid + "_") else variant
        vp = f"{SEASON_ROOT}/{cid}/{cid}_{key}.png"
        hero = f"{CAST_ROOT}/{cid}/{cid}_hero.png"
        item = {"cid": cid, "variant": variant, "orig": r.get("verdict"),
                "raw": r.get("raw"), "src": os.path.basename(srcf)}

        if not os.path.exists(vp) or not os.path.exists(hero):
            item["final"] = "缺文件"
            out.append(item)
            continue

        # 1) 嵌入相似度
        try:
            ev, eh = embed(vp), embed(hero)
            item["sim"] = round(float(np.dot(ev, eh)), 4) if ev is not None and eh is not None else None
        except Exception as ex:
            item["sim"] = None
            item["sim_err"] = str(ex)[:100]

        # 2) 第三票（正序 + 逆序各一次，取一致者）
        try:
            t1 = third_vote(hero, vp)
            t2 = third_vote(vp, hero)
            item["third"] = [t1, t2]
        except Exception as ex:
            item["third"] = ["ERR", str(ex)[:80]]

        # 仲裁规则
        sim = item.get("sim")
        t1, t2 = item["third"]
        votes_for_hero = sum(1 for t in (t1, t2) if t == "1")
        votes_for_var = sum(1 for t in (t1, t2) if t == "2")
        if sim is not None and sim >= args.sim_thresh:
            item["final"] = "同人"
            item["reason"] = f"embed_sim={sim} >= {args.sim_thresh}"
        elif votes_for_hero >= 1 and votes_for_var >= 1:
            # 第三票也平票 -> 保守判存疑
            item["final"] = "存疑"
            item["reason"] = f"embed_sim={sim}, third split"
        elif votes_for_var == 2:
            item["final"] = "换人"
            item["reason"] = f"embed_sim={sim}, third says variant"
        elif votes_for_hero == 2:
            item["final"] = "同人"
            item["reason"] = f"embed_sim={sim}, third says hero"
        else:
            item["final"] = "存疑"
            item["reason"] = f"embed_sim={sim}, third inconclusive"
        out.append(item)
        print(f"  {n+1}/{len(suspects)} {cid}_{key}: sim={item.get('sim')} "
              f"third={item.get('third')} -> {item['final']}", flush=True)

    of = files[0]
    outp = os.path.splitext(of)[0] + ".arbitrated.json"
    json.dump(out, open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    same = sum(1 for x in out if x["final"] == "同人")
    diff = sum(1 for x in out if x["final"] == "换人")
    sus = sum(1 for x in out if x["final"] == "存疑")
    print(f"[done] 同人={same} 换人={diff} 存疑={sus} ({time.time()-t0:.0f}s) -> {outp}",
          flush=True)


if __name__ == "__main__":
    main()
