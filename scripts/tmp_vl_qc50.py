#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""50 人角色库形象质检（InsightFace 粗筛 + Qwen2.5-VL 精判）：
   1) 同角色多视图是否同一人：VL 双图比对
   2) 不同角色是否撞脸：InsightFace 粗筛出高相似对 -> VL 确认
"""
import base64, glob, json, os, random, re, sys
import urllib.request

VL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen2.5-vl-72b"
CAST_ROOT = "/ssd2/duanju/characters/cast"
OUT = "/ssd2/duanju/characters/qc_report.json"
random.seed(20261114)


def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def ask(prompt, imgs):
    content = [{"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64(p)}"}} for p in imgs]
    content.append({"type": "text", "text": prompt})
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}],
                       "max_tokens": 120, "temperature": 0.0}).encode()
    req = urllib.request.Request(VL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def same_face(a, b):
    t = ask("这两张图片里是同一个人吗？只回答\"是\"或\"否\"。", [a, b]).strip()
    m = re.search(r"[是否]", t)
    return (m.group(0) if m else "?")


def main():
    cast = json.load(open("/ssd2/duanju/characters/cast.json", encoding="utf-8"))
    ids = [c["id"] for c in cast]
    heroes = {i: f"{CAST_ROOT}/{i}/{i}_hero.png" for i in ids
              if os.path.exists(f"{CAST_ROOT}/{i}/{i}_hero.png")}

    # ---------- 1) 同角色视图一致性（VL） ----------
    print("=== 1) 同角色多视图一致性 (VL) ===", flush=True)
    same_r = []
    for cid in ids:
        d = f"{CAST_ROOT}/{cid}"
        views = sorted(glob.glob(f"{d}/{cid}_*.png"))
        if len(views) < 2:
            continue
        others = [v for v in views if v != heroes.get(cid)]
        picks = random.sample(others, min(2, len(others)))
        verdicts = [{"view": os.path.basename(v), "same": same_face(heroes[cid], v)} for v in picks]
        ok = all(x["same"] == "是" for x in verdicts)
        same_r.append({"id": cid, "ok": ok, "checks": verdicts})
        print(f"  {cid}: {'PASS' if ok else 'CHECK'} {[x['same'] for x in verdicts]}", flush=True)

    # ---------- 2) 撞脸检测（InsightFace 粗筛 -> VL 精判） ----------
    print("=== 2) 撞脸检测 ===", flush=True)
    dup = []
    try:
        sys.path.insert(0, "/ssd2/short_drama/dh_scripts")
        import face_check
        print("  InsightFace 粗筛...", flush=True)
        # 用 face_check 的 CLI 风格逐对算相似度
        import subprocess
        emb = {}
        for cid, p in heroes.items():
            r = subprocess.run(["/ssd2/miniconda3/envs/videogen/bin/python",
                                "face_check.py", "--ref", p, p],
                               cwd="/ssd2/short_drama/dh_scripts",
                               capture_output=True, text=True, timeout=120)
            for line in r.stdout.splitlines():
                if line.strip().startswith("{"):
                    emb[cid] = json.loads(line).get("sim")
                    break
        print(f"  嵌入完成 {len(emb)}", flush=True)
        pairs = [(a, b) for i, a in enumerate(sorted(emb)) for b in sorted(emb)[i + 1:]
                 if emb.get(a) and emb.get(b) and emb[a] > 0.55 and emb[b] > 0.55]
        print(f"  高相似候选对: {len(pairs)}", flush=True)
        for a, b in pairs:
            s = same_face(heroes[a], heroes[b])
            tag = "DUP" if s == "是" else "ok"
            print(f"    {a} vs {b}: sim={emb[a]:.2f}/{emb[b]:.2f} -> {s} {tag}", flush=True)
            if s == "是":
                dup.append({"a": a, "b": b, "sim": [emb[a], emb[b]]})
    except Exception as ex:
        print(f"  InsightFace 不可用({ex})，改全量 VL 比对", flush=True)
        al = sorted(heroes)
        for i, a in enumerate(al):
            for b in al[i + 1:]:
                if same_face(heroes[a], heroes[b]) == "是":
                    dup.append({"a": a, "b": b})
                    print(f"    [DUP] {a} vs {b}", flush=True)

    report = {"same_view_check": same_r,
              "same_view_pass": sum(1 for x in same_r if x["ok"]),
              "same_view_total": len(same_r),
              "dup_pairs": dup}
    json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"[done] 一致性 {report['same_view_pass']}/{report['same_view_total']}, "
          f"撞脸 {len(dup)} 对 -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
