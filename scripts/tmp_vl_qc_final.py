#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最终质检：双盲交叉验证 + InsightFace 嵌入佐证。

单次 VL 判断有随机性（尤其"是/否"二选一容易偏向某个方向），
因此对每一对做两次独立询问：
  正向：先 A 后 B
  反向：先 B 后 A
两次结论一致才采信；不一致标记为"存疑"并单独列出人工复核。

输出：/ssd2/duanju/characters/qc_report_final.json
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
OUT = sys.argv[1] if len(sys.argv) > 1 else "/ssd2/duanju/characters/qc_report_final.json"
random.seed(20261114)

# 参与一致性核验的视图（back 背面无脸，turnaround 是三视图合成图，都不可直接比对）
COMPARE_VIEWS = ["headshot", "threequarter", "profile", "fullbody", "smile", "angry", "sad"]
DUP_SIM_CANDIDATE = 0.62


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


PROMPT = """这两张照片里的人物，需要你判断是不是同一个人。

请对两张图做**逐项比对**（直接对比，不要分别描述后再比）：
1) 先看第二张图能否看清面部。若第二张是背面、侧后方或五官被遮挡看不清，**回答"不可比"**。
2) 能看清则逐项对比：下颌线与下颌角形状、眉形与眉峰位置、眼型/眼距/双眼皮走向、
   鼻梁宽窄与鼻头形状、嘴唇厚薄与嘴角走向、颧骨宽窄、下巴尖圆、特殊标记（痣疤酒窝）。
3) 以上各项全部吻合才是同一人；有 2 项以上明显不同就不是同一人。

注意：表情不同不代表换人，只看面部结构；角度、远近、光线不算差异；
两人都"椭圆脸杏仁眼"是常见巧合，必须找具体差异点（鼻头形状、眼距、眉峰位置、痣的位置）。

只回答三个词之一：是 / 否 / 不可比
"""


def verdict(pa, pb):
    t = ask(PROMPT, [pa, pb], 20).strip()
    if "不可比" in t:
        return "不可比"
    m = re.search(r"(是|否)", t)
    if not m:
        return "?"
    return m.group(1)


def cross(pa, pb):
    """正向 + 反向两次判断，返回 (结论, 两次原始结果)。"""
    v1 = verdict(pa, pb)
    v2 = verdict(pb, pa)
    if v1 == v2:
        return v1, [v1, v2]
    if "不可比" in (v1, v2):
        return "不可比", [v1, v2]
    return "存疑", [v1, v2]


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

    # ---------- 1) 撞脸检测 ----------
    print("=== 1) 撞脸检测（双盲交叉 + 嵌入佐证） ===", flush=True)
    t0 = time.time()
    emb = {}
    for n, cid in enumerate(sorted(heroes)):
        e = embed(heroes[cid])
        if e is not None:
            emb[cid] = e
        if (n + 1) % 50 == 0:
            print(f"  embedded {n+1}/{len(heroes)} ({time.time()-t0:.0f}s)", flush=True)
    hids = sorted(emb)
    cand = []
    for i, a in enumerate(hids):
        for b in hids[i + 1:]:
            s = float(np.dot(emb[a], emb[b]))
            if s >= DUP_SIM_CANDIDATE:
                cand.append((a, b, s))
    cand.sort(key=lambda x: -x[2])
    print(f"  候选对 {len(cand)}", flush=True)
    dup, uncertain = [], []
    for a, b, s in cand:
        v, raw = cross(heroes[a], heroes[b])
        print(f"    {a} vs {b}: sim={s:.3f} -> {v} {raw}", flush=True)
        if v == "是":
            dup.append({"a": a, "b": b, "sim": round(s, 3), "votes": raw})
        elif v == "存疑":
            uncertain.append({"a": a, "b": b, "sim": round(s, 3), "votes": raw})

    # ---------- 2) 同角色视图一致性 ----------
    print("=== 2) 同角色视图一致性（双盲交叉） ===", flush=True)
    same_r = []
    for cid in ids:
        if cid not in heroes:
            continue
        d = f"{CAST_ROOT}/{cid}"
        checks = []
        for v in COMPARE_VIEWS:
            p = f"{d}/{cid}_{v}.png"
            if not os.path.exists(p):
                continue
            res, raw = cross(heroes[cid], p)
            checks.append({"view": v, "verdict": res, "votes": raw})
        n_ok = sum(1 for c in checks if c["verdict"] == "是")
        n_na = sum(1 for c in checks if c["verdict"] == "不可比")
        n_q = sum(1 for c in checks if c["verdict"] == "存疑")
        n_no = sum(1 for c in checks if c["verdict"] == "否")
        # 判定：可比的视图中"否"为不一致，"存疑"单独标记
        ok = (n_no == 0 and n_q == 0)
        same_r.append({"id": cid, "ok": ok, "checks": checks,
                       "same": n_ok, "na": n_na, "uncertain": n_q, "diff": n_no})
        flag = 'PASS' if ok else ('CHECK' if n_q else 'DIFF')
        print(f"  {cid}: {flag} 是{n_ok}/否{n_no}/不可比{n_na}/存疑{n_q}", flush=True)

    report = {
        "roles_total": len(ids), "roles_with_hero": len(heroes),
        "dup_confirmed": dup,
        "dup_uncertain": uncertain,
        "dup_candidates": [{"a": a, "b": b, "sim": round(s, 3)} for a, b, s in cand],
        "same_view_pass": sum(1 for x in same_r if x["ok"]),
        "same_view_total": len(same_r),
        "same_view_diff": [x["id"] for x in same_r if not x["ok"] and not x["uncertain"]],
        "same_view_uncertain": [x["id"] for x in same_r if x["uncertain"]],
        "same_view_detail": same_r,
        "method": "InsightFace 嵌入粗筛 -> VL 双盲交叉询问（正反两次一致才采信）",
    }
    json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[done] 一致性 {report['same_view_pass']}/{report['same_view_total']}, "
          f"确认撞脸 {len(dup)} 对, 存疑 {len(uncertain)} 对 -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
