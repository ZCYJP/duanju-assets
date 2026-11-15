#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""季节/节气变体形象质检：变体图 vs 该角色 hero 主形象。

变体生成的核心风险是"换了季节和场景，脸也跟着变了"（identity drift）。
因此对每张变体图与同一角色的 hero 做双盲交叉比对：
  正向：先 hero 后变体
  反向：先变体后 hero
两次结论一致才采信；不一致标"存疑"单独列出。

用法：
  python3 tmp_vl_qc_season.py                 # 全量
  python3 tmp_vl_qc_season.py --limit 300    # 只抽前300张
  python3 tmp_vl_qc_season.py --roles M01,F02
输出：/ssd2/duanju/seasons/qc_report_season.json
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

VL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen2.5-vl-72b"
SEASON_ROOT = "/ssd2/duanju/seasons"
CAST_ROOT = "/ssd2/duanju/characters/cast"
OUT = "/ssd2/duanju/seasons/qc_report_season.json"

PROMPT = """这两张照片里的人物，需要你判断是不是同一个人。

请对两张图做**逐项比对**（直接对比，不要分别描述后再比）：
1) 先看第二张图能否看清面部。若第二张是背面、侧后方或五官被遮挡看不清，**回答"不可比"**。
2) 能看清则逐项对比：下颌线与下颌角形状、眉形与眉峰位置、眼型/眼距/双眼皮走向、
   鼻梁宽窄与鼻头形状、嘴唇厚薄与嘴角走向、颧骨宽窄、下巴尖圆、特殊标记（痣疤酒窝）。
3) 以上各项全部吻合才是同一人；有 2 项以上明显不同就不是同一人。

注意：表情不同不代表换人，只看面部结构；角度、远近、光线、季节背景、
服装款式均不算差异；两人都"椭圆脸杏仁眼"是常见巧合，必须找具体差异点
（鼻头形状、眼距、眉峰位置、痣的位置）。

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


def verdict(pa, pb):
    t = ask(PROMPT, [pa, pb], 20).strip()
    if "不可比" in t:
        return "不可比"
    m = re.search(r"(是|否)", t)
    return m.group(1) if m else "?"


def cross(pa, pb):
    v1 = verdict(pa, pb)
    v2 = verdict(pb, pa)
    if v1 == v2:
        return v1, [v1, v2]
    if "不可比" in (v1, v2):
        return "不可比", [v1, v2]
    return "存疑", [v1, v2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--roles", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    only_roles = set(x.strip() for x in args.roles.split(",") if x.strip())

    items = []
    for p in sorted(glob.glob(f"{SEASON_ROOT}/*/*.png")):
        cid = os.path.basename(os.path.dirname(p))
        if only_roles and cid not in only_roles:
            continue
        hero = f"{CAST_ROOT}/{cid}/{cid}_hero.png"
        if not os.path.exists(hero):
            continue
        items.append((cid, os.path.basename(p)[:-4], p, hero))

    if args.of > 1:
        items = [x for i, x in enumerate(items) if i % args.of == args.shard]

    if args.limit:
        items = items[:args.limit]

    print(f"[data] {len(items)} variants to check "
          f"(shard {args.shard}/{args.of})", flush=True)

    report = []
    t0 = time.time()
    for n, (cid, name, vp, hero) in enumerate(items):
        try:
            v, raw = cross(hero, vp)
        except Exception as ex:
            v, raw = "ERROR", [str(ex)[:120]]
        report.append({"cid": cid, "variant": name, "verdict": v, "raw": raw})
        if (n + 1) % 20 == 0:
            el = time.time() - t0
            eta = el / (n + 1) * (len(items) - n - 1)
            print(f"  {n+1}/{len(items)} ({el:.0f}s, eta {eta/60:.1f}m)", flush=True)
            json.dump(report, open(args.out, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)

    json.dump(report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    same = sum(1 for r in report if r["verdict"] == "是")
    diff = sum(1 for r in report if r["verdict"] == "否")
    sus = sum(1 for r in report if r["verdict"] == "存疑")
    nc = sum(1 for r in report if r["verdict"] in ("不可比", "?", "ERROR"))
    print(f"[done] 同人={same} 换人={diff} 存疑={sus} 不可比/异常={nc} "
          f"总计={len(report)} -> {args.out}", flush=True)
    print("DIFF:", [f"{r['cid']}_{r['variant']}" for r in report if r["verdict"] == "否"][:60],
          flush=True)
    print("SUS:", [f"{r['cid']}_{r['variant']}" for r in report if r["verdict"] == "存疑"][:60],
          flush=True)


if __name__ == "__main__":
    main()
