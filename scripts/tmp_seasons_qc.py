#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""形象变体质量抽样核查（只读，不修改任何原图）

对 /ssd2/duanju/seasons/ 抽样（>=60 张，覆盖多角色多变体）：
  - 文件完整性 / PIL 可打开性 / 尺寸一致性 (832x1248)
  - 灰度直方图熵、平均亮度 -> 检测全黑/全白/纯色/低熵异常
  - aHash 感知哈希 -> 检测重复图 / 近重复图
输出 /ssd2/duanju/seasons_qc.json
"""
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = "/ssd2/duanju/seasons"
OUT_JSON = "/ssd2/duanju/seasons_qc.json"
EXPECT_SIZE = (832, 1248)

# 必覆盖的变体
MUST_VARIANTS = ["winter", "summer", "dongzhi", "xiazhi", "lichun"]
# 二十四节气
SOLAR = ["lichun", "yushui", "jingzhe", "chunfen", "qingming", "lixia", "xiaoman",
         "mangzhong", "xiazhi", "xiaoshu", "liqiu", "shuangjiang", "lidong",
         "daxue", "dongzhi", "xiaohan", "hanlu", "lixia2"]

VARIANTS = ["winter", "summer", "lichun", "yushui", "jingzhe", "chunfen", "qingming",
            "lixia", "xiaoman", "mangzhong", "xiazhi", "xiaoshu", "liqiu", "shuangjiang",
            "lidong", "daxue", "dongzhi", "xiaohan", "hanlu", "guyu", "lizhi", "chushu",
            "bailu", "qiufen", "hanlu2", "shuangjiang2"]

# 异常阈值
ENTROPY_LOW = 2.0        # 灰度熵低于此值视为近纯色
ENTROPY_MIN = 3.0        # 可疑低熵
MEAN_BLACK = 8.0         # 平均亮度低于此值视为近全黑
MEAN_WHITE = 247.0       # 平均亮度高于此值视为近全白
STD_LOW = 2.0            # 灰度标准差低于此值视为纯色


def shannon_entropy(gray):
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def ahash(gray, size=8):
    im = Image.fromarray(gray).resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(im, dtype=np.float64)
    return (a[:, 1:] > a[:, :-1]).flatten()


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def check_one(path):
    rec = {"file": os.path.relpath(path, ROOT), "path": path}
    try:
        with Image.open(path) as im:
            im.load()
            W, H = im.size
            rec["size"] = [W, H]
            rec["size_ok"] = (W, H) == EXPECT_SIZE
            if im.mode != "L":
                g = im.convert("L")
            else:
                g = im
            gray = np.asarray(g, dtype=np.uint8)
            rec["format"] = im.format
            rec["mode"] = im.mode
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["ok"] = False
        return rec

    rec["mean"] = round(float(gray.mean()), 3)
    rec["std"] = round(float(gray.std()), 3)
    rec["entropy"] = round(shannon_entropy(gray), 4)
    rec["ahash"] = ahash(gray).astype(np.uint8).tobytes().hex()

    # 纯色检测：唯一灰度级数
    rec["n_levels"] = int(np.count_nonzero(np.bincount(gray.reshape(-1), minlength=256)))
    rec["ok"] = True

    issues = []
    if not rec["size_ok"]:
        issues.append(f"尺寸异常({W}x{H})")
    if rec["mean"] <= MEAN_BLACK:
        issues.append(f"近全黑(mean={rec['mean']})")
    if rec["mean"] >= MEAN_WHITE:
        issues.append(f"近全白(mean={rec['mean']})")
    if rec["std"] <= STD_LOW or rec["n_levels"] <= 3:
        issues.append(f"纯色图(levels={rec['n_levels']},std={rec['std']})")
    elif rec["entropy"] < ENTROPY_LOW:
        issues.append(f"近纯色(entropy={rec['entropy']})")
    elif rec["entropy"] < ENTROPY_MIN:
        issues.append(f"低熵可疑(entropy={rec['entropy']})")
    rec["issues"] = issues
    rec["anomalous"] = bool(issues)
    return rec


def build_sample():
    cids = sorted(d for d in os.listdir(ROOT)
                  if os.path.isdir(os.path.join(ROOT, d)))
    by_cid = {}
    for cid in cids:
        files = sorted(f for f in os.listdir(os.path.join(ROOT, cid))
                       if f.lower().endswith(".png"))
        by_cid[cid] = files

    # 1) 每个变体至少抽 4 个角色 -> 覆盖所有变体
    per_variant = defaultdict(list)
    for cid, files in by_cid.items():
        for f in files:
            v = f[len(cid) + 1:-4] if f.startswith(cid + "_") else os.path.splitext(f)[0]
            per_variant[v].append(os.path.join(ROOT, cid, f))

    sample = set()
    for v, paths in sorted(per_variant.items()):
        step = max(1, len(paths) // 4)
        for i in range(0, len(paths), step):
            sample.add(paths[i])
            if len([p for p in sample]) > 0 and i // step >= 3:
                break

    # 2) 必覆盖变体额外多抽
    for v in MUST_VARIANTS:
        paths = per_variant.get(v, [])
        if not paths:
            continue
        step = max(1, len(paths) // 8)
        for i in range(0, len(paths), step):
            sample.add(paths[i])
            if len([p for p in per_variant[v][::step][:8]]) >= 8:
                break

    # 3) 每个角色至少抽 2 张（保证角色覆盖广度）
    for cid, files in sorted(by_cid.items()):
        if len(files) >= 2:
            sample.add(os.path.join(ROOT, cid, files[0]))
            sample.add(os.path.join(ROOT, cid, files[len(files) // 2]))

    return sorted(sample), per_variant, cids


def main():
    sample, per_variant, cids = build_sample()
    print(f"[qc] 角色目录 {len(cids)} 个, 变体 {len(per_variant)} 种, "
          f"抽样 {len(sample)} 张", flush=True)
    if len(sample) < 60:
        print(f"[warn] 样本不足 60, 实际 {len(sample)}", flush=True)

    results = []
    for i, p in enumerate(sample):
        rec = check_one(p)
        results.append(rec)
        if i % 20 == 0:
            print(f"  ...{i+1}/{len(sample)}", flush=True)

    # 重复检测
    by_hash = defaultdict(list)
    for r in results:
        if r.get("ahash"):
            by_hash[r["ahash"]].append(r["file"])
    exact_dupes = {h: fs for h, fs in by_hash.items() if len(fs) > 1}

    # 近重复（ahash 汉明距离 <= 4）
    hashes = [(r["file"], r["ahash"]) for r in results if r.get("ahash")]
    near = []
    for i in range(len(hashes)):
        f1, h1 = hashes[i]
        a1 = np.unpackbits(np.frombuffer(bytes.fromhex(h1), dtype=np.uint8))
        for j in range(i + 1, len(hashes)):
            f2, h2 = hashes[j]
            a2 = np.unpackbits(np.frombuffer(bytes.fromhex(h2), dtype=np.uint8))
            d = int(np.count_nonzero(a1 != a2))
            if d <= 4:
                near.append({"a": f1, "b": f2, "hamming": d})
    near.sort(key=lambda x: x["hamming"])

    anomalies = [r for r in results if r.get("anomalous") or r.get("error")]
    by_var = defaultdict(lambda: {"n": 0, "anom": 0})
    for r in results:
        v = os.path.basename(r["file"]).split("_", 1)[-1][:-4]
        by_var[v]["n"] += 1
        if r.get("anomalous") or r.get("error"):
            by_var[v]["anom"] += 1

    out = {
        "meta": {
            "root": ROOT, "expected_size": list(EXPECT_SIZE),
            "total_cid_dirs": len(cids), "total_png": sum(len(v) for v in per_variant.values()),
            "variants": sorted(per_variant.keys()),
            "sample_size": len(results),
            "thresholds": {"entropy_low": ENTROPY_LOW, "entropy_min": ENTROPY_MIN,
                           "mean_black": MEAN_BLACK, "mean_white": MEAN_WHITE,
                           "std_low": STD_LOW},
        },
        "summary": {
            "sampled": len(results),
            "ok": sum(1 for r in results if r.get("ok")),
            "broken": sum(1 for r in results if r.get("error")),
            "size_mismatch": sum(1 for r in results if r.get("ok") and not r.get("size_ok")),
            "anomalous": len(anomalies),
            "exact_duplicate_groups": len(exact_dupes),
            "near_duplicate_pairs": len(near),
            "by_variant": {k: v for k, v in sorted(by_var.items())},
        },
        "anomalies": anomalies,
        "exact_duplicates": [{"hash": h, "files": fs} for h, fs in exact_dupes.items()],
        "near_duplicates": near[:100],
        "items": results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    s = out["summary"]
    print(f"\n[summary] 抽样 {s['sampled']} | 损坏 {s['broken']} | "
          f"尺寸不符 {s['size_mismatch']} | 异常 {s['anomalous']} | "
          f"完全重复组 {s['exact_duplicate_groups']} | 近重复对 {s['near_duplicate_pairs']}",
          flush=True)
    if anomalies:
        print("[anomalies]")
        for a in anomalies:
            print("  ", a.get("file"), a.get("issues") or a.get("error"), flush=True)
    if near:
        print("[near dup] top5:", near[:5], flush=True)
    print(f"[done] -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
