"""用独立真值（谐波阶梯）+ 性别先验，裁决 150 个基线音色的绝对八度。

背景
----
v13/v14 的达标判据是「窄窗读数」，窄窗以 target 为中心 ±15%，
因此存在假达标。真正的风险在 **scale（基线尺度）的绝对八度**：
scale 错 2 倍 => 目标错 2 倍 => 成品音高跑出该性别音域（性别漂移）。

本脚本做三件事（互不循环）：
1. 用谐波阶梯法独立测出每个基线音频的绝对 f0（不看任何 target）
2. 用角色库的 gender 字段做先验校验：
   男声 f0 典型 85~185Hz，女声典型 165~255Hz
   -> 超出该区间即为八度错误，输出修正因子
3. 与现有 base_f0_v14.json 对比，报告差异

输出
----
  /ssd2/duanju/voices/base_f0_truth.json
    {cid: {"f0": 绝对Hz, "factor": 相对v14的八度修正, "gender":..., "src":...}}
"""
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tmp_f0_truth import measure_truth  # noqa: E402

VOICES = "/ssd2/duanju/voices"
CAST = "/ssd2/duanju/scripts/cast_all_150.json"

# 性别先验区间（Hz），来自真实普通话人声分布
GENDER_RANGE = {
    "男": (85.0, 185.0),
    "女": (165.0, 255.0),
}


def find_base_file(cid):
    for pat in (f"{VOICES}/samples/{cid}_*_voice.wav",
                f"{VOICES}/samples_ext/{cid}_*_voice.wav"):
        g = sorted(glob.glob(pat))
        if g:
            return g[0]
    return ""


def one(cid):
    p = find_base_file(cid)
    if not p:
        return cid, {"f0": 0.0, "path": "", "note": "文件缺失"}
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    r = measure_truth(x, sr)
    return cid, {"f0": r["f0"], "d": r["ladder_d"], "score": r["score"],
                 "h1": r["h1"], "hit": r["hit"], "n_peaks": r["n_peaks"],
                 "oct_ev": r["oct_ev"], "path": p,
                 "peaks": r["peaks"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{VOICES}/base_f0_truth.json")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    cast = json.load(open(CAST))
    cids = [c["id"] for c in cast]
    gender = {c["id"]: c.get("gender", "") for c in cast}

    print(f"[1/3] 独立测量 {len(cids)} 个基线音色（谐波阶梯法）...")
    res = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for cid, r in ex.map(one, cids, chunksize=2):
            res[cid] = r

    # 与 v14 对比 + 性别先验裁决
    try:
        v14 = json.load(open(f"{VOICES}/base_f0_v14.json"))
    except Exception:
        v14 = {}
    # v14 结构兼容：{cid: value} 或 {cid: {"f0":...}}
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

    print("[2/3] 性别先验校验 + 与 v14 对比")
    out = {}
    stats = {"ok": 0, "oct": 0, "out_of_range": 0, "no_read": 0}
    details = []
    for cid in cids:
        r = res[cid]
        g = gender.get(cid, "")
        lo, hi = GENDER_RANGE.get(g, (60.0, 300.0))
        f0 = r["f0"]
        old = getf(v14, cid)
        factor = 1.0
        note = "ok"
        if f0 <= 0:
            note = "无读数"
            stats["no_read"] += 1
        elif f0 < lo / 1.10:
            # 太低：可能是低了一个八度 -> 乘 2
            factor = 2.0
            note = f"低于{g}声下限({lo:.0f})，疑低八度"
            stats["oct"] += 1
        elif f0 > hi * 1.10:
            # 太高：可能是高了一个八度 -> 除 2
            factor = 0.5
            note = f"高于{g}声上限({hi:.0f})，疑高八度"
            stats["oct"] += 1
        else:
            stats["ok"] += 1
        out[cid] = {
            "f0": round(f0, 2),
            "ladder_d": r.get("d", 0.0),
            "score": r.get("score", 0.0),
            "h1": r.get("h1", 0.0),
            "n_peaks": r.get("n_peaks", 0),
            "oct_ev": r.get("oct_ev", 0.0),
            "gender": g,
            "range": [lo, hi],
            "v14": round(old, 2),
            "factor": factor,
            "note": note,
            "peaks": r.get("peaks", []),
        }
        if factor != 1.0:
            details.append((cid, g, round(f0, 1), round(old, 1), factor, note))

    print(f"[3/3] 写出 {args.out}")
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=1)

    print("=" * 76)
    print("裁决结果")
    print("=" * 76)
    print(f"  正常 {stats['ok']}  八度疑似错误 {stats['oct']}  "
          f"超区间 {stats['out_of_range']}  无读数 {stats['no_read']}")
    if details:
        print("-" * 76)
        print(f"{'cid':<7}{'性别':<6}{'独立f0':>9}{'v14尺度':>10}{'修正':>7}  说明")
        for cid, g, f0, old, fac, note in details:
            print(f"{cid:<7}{g:<6}{f0:>9.1f}{old:>10.1f}{fac:>7.2f}  {note}")

    # 对比统计
    ratios = []
    for cid in cids:
        o = out[cid]
        if o["v14"] > 0 and o["f0"] > 0:
            ratios.append(o["f0"] / o["v14"])
    if ratios:
        ratios = np.array(ratios)
        print("-" * 76)
        print(f"独立f0 / v14尺度  中位 {np.median(ratios):.3f}  "
              f"p10 {np.percentile(ratios,10):.3f}  p90 {np.percentile(ratios,90):.3f}")
        hist = {}
        for r in ratios:
            k = round(r, 1)
            hist[k] = hist.get(k, 0) + 1
        print("倍率直方图:", dict(sorted(hist.items())))


if __name__ == "__main__":
    main()
