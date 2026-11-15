#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""角色库形象质检（InsightFace 嵌入 + Qwen2.5-VL 精判）。

修复要点：
  1) 原脚本用 `face_check.py --ref p p`（图片与自己比），sim 恒为 1.0，
     导致所有配对都进入 VL 判定；现改为一次性提取每张图的人脸嵌入，
     两两做余弦相似度，真正筛出高相似候选对。
  2) VL 提示词改为严格判据（要求五官/脸型/发型完全一致才答"是"），
     避免把"看起来像"的不同人误判为同一人。
  3) 一致性检查同样先走嵌入粗筛，只有可疑对才送 VL，减少调用量。

输出：/ssd2/duanju/characters/qc_report_<tag>.json
"""
import base64, glob, json, os, random, re, sys, warnings
import urllib.request
import numpy as np
import cv2
import insightface

warnings.filterwarnings('ignore')
sys.path.insert(0, "/ssd2/short_drama/dh_scripts")

VL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen2.5-vl-72b"
CAST_ROOT = "/ssd2/duanju/characters/cast"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/ssd2/duanju/characters/qc_report.json"
TAG = os.path.basename(OUT).replace(".json", "")
random.seed(20261114)

# 阈值：同一人不同视角 cos>=0.55（编辑模型保脸通常 0.6~0.8）；
#       不同角色若 cos>=0.62 才可能是撞脸，需 VL 精判
VIEW_SIM_SUSPECT = 0.55
DUP_SIM_CANDIDATE = 0.62


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
    """提取最大人脸的归一化嵌入；无脸返回 None。"""
    global _app
    if _app is None:
        # 优先用 GPU（CPU provider 在本机会僵住）；无 CUDA 时回退 CPU
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            import torch
            if not torch.cuda.is_available():
                providers = ["CPUExecutionProvider"]
        except Exception:
            providers = ["CPUExecutionProvider"]
        _app = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
        _app.prepare(ctx_id=0 if providers[0].startswith("CUDA") else -1, det_size=(640, 640))
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


def cos(a, b):
    return float(np.dot(a, b))


def main():
    cast_files = ["/ssd2/duanju/scripts/cast_all_150.json"]
    cast = []
    for cf in cast_files:
        cast += json.load(open(cf, encoding="utf-8"))
    ids = [c["id"] for c in cast]
    heroes = {i: f"{CAST_ROOT}/{i}/{i}_hero.png" for i in ids
              if os.path.exists(f"{CAST_ROOT}/{i}/{i}_hero.png")}
    print(f"[data] {len(ids)} roles, {len(heroes)} heroes present", flush=True)

    # ---------- 提取所有嵌入 ----------
    print("=== 0) InsightFace 提取嵌入 ===", flush=True)
    hero_emb = {}
    view_emb = {}   # cid -> {view_name: emb}
    for cid in heroes:
        e = embed(heroes[cid])
        if e is not None:
            hero_emb[cid] = e
        d = f"{CAST_ROOT}/{cid}"
        for v in sorted(glob.glob(f"{d}/{cid}_*.png")):
            if v == heroes[cid]:
                continue
            ev = embed(v)
            if ev is not None:
                view_emb.setdefault(cid, {})[os.path.basename(v)] = ev
    print(f"  hero emb {len(hero_emb)}, view emb {sum(len(v) for v in view_emb.values())}", flush=True)

    # ---------- 1) 同角色视图一致性 ----------
    print("=== 1) 同角色多视图一致性 ===", flush=True)
    same_r = []
    for cid in sorted(heroes):
        if cid not in view_emb:
            continue
        he = hero_emb.get(cid)
        checks = []
        for vn, ve in sorted(view_emb[cid].items()):
            if he is None:
                continue
            s = cos(he, ve)
            # 嵌入明显低（<0.35）或高（>0.93，多脸/贴图异常）才需要 VL 复核
            if s < VIEW_SIM_SUSPECT or s > 0.93:
                v = same_face_strict(heroes[cid], f"{CAST_ROOT}/{cid}/{vn}")
                checks.append({"view": vn, "sim": round(s, 3), "vl": v})
            else:
                checks.append({"view": vn, "sim": round(s, 3), "vl": "是"})
        ok = all(c["vl"] == "是" for c in checks)
        low = [c for c in checks if c["sim"] < 0.35]
        same_r.append({"id": cid, "ok": ok and not low, "checks": checks,
                       "min_sim": round(min((c["sim"] for c in checks), default=0), 3)})
        flag = 'PASS' if (ok and not low) else 'CHECK'
        print(f"  {cid}: {flag} min_sim={same_r[-1]['min_sim']}", flush=True)

    # ---------- 2) 撞脸检测 ----------
    print("=== 2) 撞脸检测（嵌入粗筛 -> VL 精判） ===", flush=True)
    hids = sorted(hero_emb)
    cand = []
    for i, a in enumerate(hids):
        for b in hids[i + 1:]:
            s = cos(hero_emb[a], hero_emb[b])
            if s >= DUP_SIM_CANDIDATE:
                cand.append((a, b, s))
    cand.sort(key=lambda x: -x[2])
    print(f"  高相似候选对: {len(cand)} (阈值 {DUP_SIM_CANDIDATE})", flush=True)
    dup = []
    for a, b, s in cand:
        v = same_face_strict(heroes[a], heroes[b])
        tag = "DUP" if v == "是" else "ok"
        print(f"    {a} vs {b}: sim={s:.3f} -> {v} {tag}", flush=True)
        if v == "是":
            dup.append({"a": a, "b": b, "sim": round(s, 3)})

    # ---------- 3) 背景文字核验 ----------
    scene_report = []
    scene_root = "/ssd2/duanju/scenes_100"
    if os.path.isdir(scene_root):
        print("=== 3) 背景招牌文字核验 (VL) ===", flush=True)
        import re as _re
        src = open("/ssd2/duanju/scripts/gen_scenes_100.py", encoding="utf-8").read()
        ns = {}
        m = _re.search(r"SCENES = \[(.*?)\n\]", src, _re.S)
        exec("SCENES = [" + m.group(1) + "\n]", ns)
        for sid, cn, en, text, lang in ns["SCENES"]:
            p = f"{scene_root}/{sid}.png"
            if not os.path.exists(p):
                scene_report.append({"sid": sid, "expected": text, "lang": lang, "status": "MISSING"})
                continue
            q = (f"这张背景图里有一个招牌。请只回答招牌上实际写出的文字，"
                 f"不要解释，不要加引号。如果招牌文字模糊或看不清，回复\"看不清\"。")
            got = ask(q, [p], maxtok=60).strip().replace("\n", " ")
            ok = (text.replace(" ", "") in got.replace(" ", "")) or \
                 (text.replace(" ", "").lower() in got.replace(" ", "").lower())
            scene_report.append({"sid": sid, "expected": text, "lang": lang,
                                 "got": got, "status": "OK" if ok else "MISMATCH"})
            if not ok:
                print(f"    [TEXT] {sid} expected={text!r} got={got!r}", flush=True)
        n_ok = sum(1 for r in scene_report if r["status"] == "OK")
        print(f"  背景文字核验: {n_ok}/{len(scene_report)} OK", flush=True)

    report = {
        "tag": TAG,
        "roles_total": len(ids), "roles_with_hero": len(heroes),
        "same_view_pass": sum(1 for x in same_r if x["ok"]),
        "same_view_total": len(same_r),
        "same_view_fail": [x["id"] for x in same_r if not x["ok"]],
        "same_view_detail": same_r,
        "dup_pairs": dup,
        "dup_candidates": [{"a": a, "b": b, "sim": round(s, 3)} for a, b, s in cand],
        "scene_text_ok": sum(1 for r in scene_report if r["status"] == "OK"),
        "scene_text_total": len(scene_report),
        "scene_text_detail": scene_report,
    }
    json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[done] 一致性 {report['same_view_pass']}/{report['same_view_total']}, "
          f"撞脸 {len(dup)} 对, 背景文字 {report['scene_text_ok']}/{report['scene_text_total']}"
          f" -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
