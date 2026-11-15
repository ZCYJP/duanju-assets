"""pyin 绝对读数 × 性别音域交叉验证（裁决 scale 绝对八度）。"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tmp_f0_arena import est_pyin  # noqa: E402

VOICES = "/ssd2/duanju/voices"
CAST = "/ssd2/duanju/scripts/cast_all_150.json"

# 真实普通话人声的典型 f0 区间（含自然变异余量）
GENDER_RANGE = {
    "男": (85.0, 185.0),
    "女": (165.0, 255.0),
}


def find_base(cid):
    for d in ("samples", "samples_ext"):
        g = sorted(glob.glob(f"{VOICES}/{d}/{cid}_*_voice.wav"))
        if g:
            return g[0]
    return ""


def one(c):
    cid = c["id"]
    p = find_base(cid)
    if not p:
        return {"cid": cid, "gender": c.get("gender", ""), "f0": 0.0, "path": ""}
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    return {"cid": cid, "gender": c.get("gender", ""),
            "f0": round(est_pyin(x, sr), 2), "path": p}


def main():
    cast = json.load(open(CAST))
    print(f"[1/3] pyin 测量 {len(cast)} 个基线音色")
    with ProcessPoolExecutor(max_workers=48) as ex:
        rows = list(ex.map(one, cast, chunksize=2))

    print("[2/3] 性别音域交叉验证")
    out = {}
    stats = {}
    for g in ("男", "女"):
        grp = [r for r in rows if r["gender"] == g]
        v = np.array([r["f0"] for r in grp])
        lo, hi = GENDER_RANGE[g]
        inr = int(((v >= lo) & (v <= hi)).sum())
        below = int((v < lo).sum())
        above = int((v > hi).sum())
        stats[g] = {"n": len(v), "in_range": inr, "below": below, "above": above,
                    "median": float(np.median(v)), "min": float(v.min()),
                    "max": float(v.max())}
        print(f"  {g}: n={len(v)}  中位 {np.median(v):.1f}  区间 {lo:.0f}~{hi:.0f}Hz  "
              f"落入 {inr}  偏低 {below}  偏高 {above}")
        for r in grp:
            out[r["cid"]] = {
                "f0_pyin": r["f0"], "gender": r["gender"],
                "range": [lo, hi], "path": r["path"],
                "in_range": bool(lo <= r["f0"] <= hi),
            }

    print("[3/3] 与现有 base_f0_v14.json 对比")
    try:
        v14 = json.load(open(f"{VOICES}/base_f0_v14.json"))
    except Exception:
        v14 = {}

    def getf(d, cid):
        v = d.get(cid)
        if isinstance(v, dict):
            for k in ("f0", "scale", "value", "base_f0"):
                if k in v:
                    return float(v[k])
            return 0.0
        try:
            return float(v)
        except Exception:
            return 0.0

    ratios = []
    print(f"{'cid':<7}{'性别':<6}{'pyin':>9}{'v14':>9}{'比值':>8}  判定")
    diffs = []
    for r in rows:
        cid = r["cid"]
        old = getf(v14, cid)
        p = r["f0"]
        if p > 0 and old > 0:
            rt = p / old
            ratios.append(rt)
            lo, hi = GENDER_RANGE.get(r["gender"], (60, 300))
            flag = "OK"
            if not (lo <= p <= hi):
                flag = "!! pyin 超出音域"
                diffs.append((cid, r["gender"], p, old, rt, flag))
            elif abs(rt - 1.0) > 0.25:
                flag = "v14 与 pyin 差 >25%"
                diffs.append((cid, r["gender"], p, old, rt, flag))
    ratios = np.array(ratios)
    print("-" * 74)
    print(f"pyin/v14 比值: 中位 {np.median(ratios):.3f}  "
          f"p10 {np.percentile(ratios,10):.3f}  p90 {np.percentile(ratios,90):.3f}")
    hist = {}
    for x in ratios:
        k = round(x, 1)
        hist[k] = hist.get(k, 0) + 1
    print("比值直方图:", dict(sorted(hist.items())))
    print(f"落在 [0.85,1.18] 的比例: {float(((ratios>=0.85)&(ratios<=1.18)).mean())*100:.1f}%")
    print(f"落在 [0.45,0.56] 或 [1.78,2.25] 的比例: "
          f"{float((((ratios>=0.45)&(ratios<=0.56))|((ratios>=1.78)&(ratios<=2.25))).mean())*100:.1f}%")

    if diffs:
        print("-" * 74)
        print("异常明细:")
        for cid, g, p, old, rt, flag in diffs:
            print(f"{cid:<7}{g:<6}{p:>9.1f}{old:>9.1f}{rt:>8.3f}  {flag}")

    outp = f"{VOICES}/base_f0_pyin_audit.json"
    json.dump({"stats": stats, "items": out}, open(outp, "w"),
              ensure_ascii=False, indent=1)
    print(f"-> {outp}")


if __name__ == "__main__":
    main()
