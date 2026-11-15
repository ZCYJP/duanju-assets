#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""150 人形象质检（轻量版，避开 InsightFace 的 CPU 瓶颈）。

原方案要跑 1500 张图的 InsightFace 嵌入，但本机 onnxruntime 只有
CPUExecutionProvider（无 CUDA），全核跑要几十分钟且会与生成任务抢 CPU。
改为：
  1) 撞脸检测 —— 只对 150 张 hero 做 InsightFace 嵌入，两两算余弦相似度。
     这正是撞脸检测需要的全部样本。
  2) 同角色视图一致性 —— 用 VL 双图判别，抽样 2 个视图/角色（与原方案同口径），
     嵌入只用于 hero 之间的互比，不做全量视图嵌入。

输出：/ssd2/duanju/characters/qc_report_150.json
"""
import base64, glob, json, os, random, re, sys, time, warnings
import urllib.request
import numpy as np
import cv2
import insightface

warnings.filterwarnings('ignore')

VL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen2.5-vl-72b"
CAST_ROOT = "/ssd2/duanju/characters/cast"
CAST_JSON = "/ssd2/duanju/scripts/cast_all_150.json"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/ssd2/duanju/characters/qc_report_150.json"
random.seed(20261114)

DUP_SIM_CANDIDATE = 0.62      # 不同角色的相似度超过此值才送 VL 精判


def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def ask(prompt, imgs, maxtok=200):
    content = [{"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64(p)}"}} for p in imgs]
    content.append({"type": "text", "text": prompt})
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}],
                       "max_tokens": maxtok, "temperature": 0.0}).encode()
    req = urllib.request.Request(VL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def same_face_strict(a, b):
    """严格判据：只有五官、脸型、发型几乎完全一致才答'是'。"""
    p = ("这是两张不同人物的照片。请严格判断画面中的人是不是同一个人。\n"
         "判断标准（必须同时满足才算同一人）：\n"
         "1) 脸型轮廓一致（下颌线、颧骨形状相同）\n"
         "2) 五官比例与位置一致（眼距、眼型、鼻梁鼻翼、嘴唇形状相同）\n"
         "3) 眉形、发际线、发型完全一致\n"
         "4) 肤色与面部特征（痣、疤、酒窝）一致\n"
         "仅仅『看起来像』『都是同一类人』『年龄性别相同』都不算同一人。\n"
         "只回答一个字符：是 或 否")
    t = ask(p, [a, b], maxtok=16).strip()
    m = re.search(r"[是否]", t)
    return (m.group(0) if m else "?")


_app = None


def embed(path):
    global _app
    if _app is None:
        _app = insightface.app.FaceAnalysis(name="buffalo_l",
                                            providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=-1, det_size=(640, 640))
    im = cv2.imread(path)
    if im is None:
        return None
    fs = [x for x in _app.get(im) if x.det_score > 0.5]
    if not fs:
        p = im.shape[0] // 4
        fs = [x for x in _app.get(cv2.copyMakeBorder(im, p, p, p, p, cv2.BORDER_CONSTANT, value=0))
              if x.det_score > 0.5]
    if not fs:
        return None
    f = max(fs, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    return f.normed_embedding


def main():
    cast = json.load(open(CAST_JSON, encoding='utf-8'))
    ids = [c['id'] for c in cast]
    heroes = {i: f"{CAST_ROOT}/{i}/{i}_hero.png" for i in ids
              if os.path.exists(f"{CAST_ROOT}/{i}/{i}_hero.png")}
    print(f"[data] {len(ids)} roles, {len(heroes)} heroes", flush=True)

    # ---------- 1) 撞脸检测（150 张 hero 嵌入，两两比对）----------
    print("=== 1) 撞脸检测 ===", flush=True)
    t0 = time.time()
    emb = {}
    for n, cid in enumerate(sorted(heroes)):
        e = embed(heroes[cid])
        if e is not None:
            emb[cid] = e
        if (n + 1) % 25 == 0:
            print(f"  embedded {n+1}/{len(heroes)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  hero emb {len(emb)} ({time.time()-t0:.0f}s)", flush=True)

    hids = sorted(emb)
    cand = []
    for i, a in enumerate(hids):
        for b in hids[i + 1:]:
            s = float(np.dot(emb[a], emb[b]))
            if s >= DUP_SIM_CANDIDATE:
                cand.append((a, b, s))
    cand.sort(key=lambda x: -x[2])
    print(f"  高相似候选对: {len(cand)} (阈值 {DUP_SIM_CANDIDATE})", flush=True)
    dup = []
    for a, b, s in cand:
        v = same_face_strict(heroes[a], heroes[b])
        print(f"    {a} vs {b}: sim={s:.3f} -> {v} {'DUP' if v == '是' else 'ok'}", flush=True)
        if v == "是":
            dup.append({"a": a, "b": b, "sim": round(s, 3)})

    # ---------- 2) 同角色视图一致性（VL 抽样 2 视图/角色）----------
    print("=== 2) 同角色多视图一致性 ===", flush=True)
    same_r = []
    for cid in ids:
        if cid not in heroes:
            continue
        d = f"{CAST_ROOT}/{cid}"
        views = sorted(glob.glob(f"{d}/{cid}_*.png"))
        if len(views) < 2:
            continue
        others = [v for v in views if v != heroes[cid]]
        picks = random.sample(others, min(2, len(others)))
        checks = [{"view": os.path.basename(v), "vl": same_face_strict(heroes[cid], v)} for v in picks]
        ok = all(c["vl"] == "是" for c in checks)
        same_r.append({"id": cid, "ok": ok, "checks": checks})
        print(f"  {cid}: {'PASS' if ok else 'CHECK'} {[c['vl'] for c in checks]}", flush=True)

    report = {
        "roles_total": len(ids), "roles_with_hero": len(heroes),
        "dup_pairs": dup,
        "dup_candidates": [{"a": a, "b": b, "sim": round(s, 3)} for a, b, s in cand],
        "same_view_pass": sum(1 for x in same_r if x["ok"]),
        "same_view_total": len(same_r),
        "same_view_fail": [x["id"] for x in same_r if not x["ok"]],
        "same_view_detail": same_r,
    }
    json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[done] 一致性 {report['same_view_pass']}/{report['same_view_total']}, "
          f"撞脸 {len(dup)} 对 -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
