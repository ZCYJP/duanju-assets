"""成品的 pyin 自校准 round-trip：确认 pyin 对成品音频的读数自洽。

逻辑
----
对成品音频 x（已按相对锚定处理）：
  施加已知降八度 -> y（真实 f0 必然减半）
  pyin(y) 应 ≈ pyin(x) / 2
若不满足，说明 pyin 在该音频上锁到了不同八度（前后不一致）
=> 该条的 pyin 读数不可信，倍率结论需降级。

这把「pyin 95.8% 八度正确率」的总体统计，落到**每一条具体音频**上。
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tmp_f0_relative_anchor import pyin_f0, pitch_shift_once, find_base  # noqa: E402
from tmp_f0_verify_rel import yin_track_median  # noqa: E402

VOICES = "/ssd2/duanju/voices"
TARGET_RATIO = {
    "xi": (1.00, 1.30), "nu": (0.95, 1.30), "ai": (0.82, 1.02),
    "le": (0.95, 1.12), "ju": (1.00, 1.30), "jing": (1.05, 1.40),
    "yan": (0.85, 1.05), "you": (0.90, 1.08), "xiu": (0.98, 1.22),
}


def selfcal(x, sr, f, factors=(0.5, 2.0)):
    """自校准：施加已知 k，检查 pyin 读数倍率。"""
    res = {}
    for k in factors:
        y = pitch_shift_once(x, sr, k)
        b = pyin_f0(y, sr)
        res[k] = round(b / f, 4) if f > 0 and b > 0 else 0.0
    # 一致性：|r/0.5 - 1| 与 |r/2 - 1| 的较小者
    e1 = abs(res[0.5] / 0.5 - 1.0) if res[0.5] > 0 else 9.9
    e2 = abs(res[2.0] / 2.0 - 1.0) if res[2.0] > 0 else 9.9
    # 允许 15% 视为自洽
    res["err"] = round(min(e1, e2), 4)
    res["consistent"] = bool(res["err"] <= 0.15)
    return res


def one(job):
    cid, emo, path, base_path = job
    try:
        x, sr = sf.read(path, dtype="float32")
        bx, bsr = sf.read(base_path, dtype="float32")
    except Exception:
        return None
    if x.ndim > 1:
        x = x[:, 0]
    if bx.ndim > 1:
        bx = bx[:, 0]
    f_e = pyin_f0(x, sr)
    f_b = pyin_f0(bx, bsr)
    if f_e <= 0 or f_b <= 0:
        return None
    r = f_e / f_b
    lo_r, hi_r = TARGET_RATIO[emo]
    sc_e = selfcal(x, sr, f_e)
    sc_b = selfcal(bx, bsr, f_b)
    return {
        "cid": cid, "emotion": emo, "r": round(r, 4),
        "lo_r": lo_r, "hi_r": hi_r,
        "in_range": bool(lo_r * 0.94 <= r <= hi_r * 1.06),
        "emo_cal": sc_e, "base_cal": sc_b,
        "emo_ok": sc_e["consistent"], "base_ok": sc_b["consistent"],
        "both_ok": bool(sc_e["consistent"] and sc_b["consistent"]),
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", default=f"{VOICES}/emotions_fixed_v16")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=f"{VOICES}/selfcal_v16.json")
    args = ap.parse_args()

    jobs = []
    for cid in sorted(os.listdir(f"{VOICES}/emotions")):
        bp = find_base(cid)
        if not bp:
            continue
        for emo in TARGET_RATIO:
            p = os.path.join(args.fixed, f"{cid}_{emo}.wav")
            if os.path.exists(p):
                jobs.append((cid, emo, p, bp))
    print(f"[1/2] 自校准 {len(jobs)} 组（每组 2 次变调 × 2 音频）")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for r in ex.map(one, jobs, chunksize=2) if r]

    print("[2/2] 统计")
    n = len(rows)
    emo_ok = sum(1 for r in rows if r["emo_ok"])
    base_ok = sum(1 for r in rows if r["base_ok"])
    both = sum(1 for r in rows if r["both_ok"])
    print(f"  成品自校准一致: {emo_ok}/{n} ({emo_ok/max(n,1)*100:.1f}%)")
    print(f"  基线自校准一致: {base_ok}/{n} ({base_ok/max(n,1)*100:.1f}%)")
    print(f"  两者都一致:      {both}/{n} ({both/max(n,1)*100:.1f}%)")
    ok = sum(1 for r in rows if r["in_range"])
    print(f"  相对倍率在区间:  {ok}/{n} ({ok/max(n,1)*100:.1f}%)")
    # 交叉：自校准一致的样本里，倍率达标率
    sub = [r for r in rows if r["both_ok"]]
    if sub:
        o2 = sum(1 for r in sub if r["in_range"])
        print(f"  在「自校准一致」子集内，倍率达标: {o2}/{len(sub)} "
              f"({o2/len(sub)*100:.1f}%)")
    # 不一致的样本
    bad = [r for r in rows if not r["both_ok"]]
    if bad:
        print("-" * 74)
        print(f"自校准不一致 {len(bad)} 条（前 15）：")
        for r in bad[:15]:
            print(f"  {r['cid']:<6}{r['emotion']:<6} r={r['r']:.3f} "
                  f"区间[{r['lo_r']},{r['hi_r']}] in={r['in_range']} "
                  f"emo_cal={r['emo_cal']} base_cal={r['base_cal']}")
    json.dump(rows, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
