#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对 QC 判"换人"的季节变体做嵌入相似度复核。

VL 双盲的"否"结论可能有系统性偏差（尤其某些场景的背景会干扰判断）。
用 InsightFace 嵌入做定量复核：
  sim >= 0.45  -> 实际是同一人，VL 误判
  sim <  0.45  -> 确实换脸，需要重生成

同时统计"换人"在各变体上的分布，找出高发变体（通常是场景干扰而非生成缺陷）。

用法：python3 tmp_qc_season_emb.py
输出：/ssd2/duanju/seasons/qc_diff_review.json
"""
import collections
import glob
import json
import os

import cv2
import numpy as np
import insightface
import warnings

warnings.filterwarnings("ignore")

SEASON_ROOT = "/ssd2/duanju/seasons"
CAST_ROOT = "/ssd2/duanju/characters/cast"
THRESH = 0.45

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


def main():
    # 收集所有"否"
    diffs = []
    for pat in ("qc_shard_*.json", "qc_gap_*.json"):
        for f in sorted(glob.glob(f"{SEASON_ROOT}/{pat}")):
            if ".arbitrated" in f:
                continue
            for x in json.load(open(f, encoding="utf-8")):
                if x.get("verdict") == "否":
                    diffs.append(x)
    # 去重
    seen = set()
    uniq = []
    for x in diffs:
        k = (x["cid"], x["variant"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)

    print(f"[data] 判否 {len(uniq)} 张，开始嵌入复核", flush=True)
    out = []
    for n, x in enumerate(uniq):
        cid = x["cid"]
        variant = x["variant"]
        key = variant.split("_", 1)[1] if variant.startswith(cid + "_") else variant
        vp = f"{SEASON_ROOT}/{cid}/{cid}_{key}.png"
        hero = f"{CAST_ROOT}/{cid}/{cid}_hero.png"
        item = {"cid": cid, "variant": key}
        try:
            ev, eh = embed(vp), embed(hero)
            if ev is None or eh is None:
                item["verdict"] = "无脸"
                item["sim"] = None
            else:
                sim = float(np.dot(ev, eh))
                item["sim"] = round(sim, 4)
                item["verdict"] = "实际同人" if sim >= THRESH else "真换脸"
        except Exception as ex:
            item["verdict"] = "错误"
            item["error"] = str(ex)[:100]
        out.append(item)
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(uniq)}", flush=True)
            json.dump(out, open(f"{SEASON_ROOT}/qc_diff_review.json", "w",
                                encoding="utf-8"), ensure_ascii=False, indent=2)

    json.dump(out, open(f"{SEASON_ROOT}/qc_diff_review.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    c = collections.Counter(x["verdict"] for x in out)
    print(f"\n[复核结果] {dict(c)}")

    by_var = collections.Counter(x["variant"] for x in out if x["verdict"] == "真换脸")
    if by_var:
        print("\n真换脸按变体分布:")
        for v, n in by_var.most_common(15):
            print(f"  {v}: {n}")
    real = [x for x in out if x["verdict"] == "真换脸"]
    if real:
        print("\n需要重生成的:")
        for x in real:
            print(f"  {x['cid']}_{x['variant']} sim={x.get('sim')}")

    # 同时统计所有变体的"否"分布，识别高发变体
    allvar = collections.Counter(x["variant"].split("_", 1)[-1]
                                 for x in uniq)
    print("\nVL判否的变体分布（前10）:")
    for v, n in allvar.most_common(10):
        print(f"  {v}: {n}")


if __name__ == "__main__":
    main()
