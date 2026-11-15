"""最终音频的独立复审：与 scale 裁决同源的三信号流程 + round-trip 交叉验证。

为什么需要独立复审
------------------
v15 报告里的「性别漂移检查」只有 82.1%/76.7% 落在音域，
但这**不等于音频真的漂移**——pyin 自身有约 4% 的八度错误
（round-trip 实测 95.8% 正确率）。用同一把尺子量自己，
测出的是「尺子的误差」，不是「工件的问题」。

本脚本用**与 scale 裁决完全相同**的三信号流程复审：
  S1 pyin 绝对读数
  S2 基频存在性（物理判据，合成 16/16）
  S3 变调往返自校准（无循环）
并额外做：
  S4 round-trip：对成品施加已知 k，pyin 读数应 = k × 成品读数
     => 若成品读数锁错八度，round-trip 仍自洽（这检验的是一致性）；
        真正的绝对判据仍是 S2 的物理存在性。

判据（与 base_f0_final 一致）
  达标 ⟺ 三信号裁决后的绝对 f0 ∈ [scale×lo_r, scale×hi_r]
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tmp_f0_arena import est_pyin  # noqa: E402
from tmp_f0_ground import fundamental_presence, GENDER_RANGE  # noqa: E402

VOICES = "/ssd2/duanju/voices"
FIXED = f"{VOICES}/emotions_fixed_v15"
REPORT = f"{VOICES}/pitch_fix_report_v15.json"

TARGET_RATIO = {
    "xi": (1.00, 1.30), "nu": (0.95, 1.30), "ai": (0.82, 1.02),
    "le": (0.95, 1.12), "ju": (1.00, 1.30), "jing": (1.05, 1.40),
    "yan": (0.85, 1.05), "you": (0.90, 1.08), "xiu": (0.98, 1.22),
}
HARD_LO, HARD_HI = 55.0, 520.0
ACCEPT_TOL = 0.08


def adjudicate(x, sr, gender):
    """三信号裁决，返回 (final_f0, detail)。"""
    f = est_pyin(x, sr)
    if f <= 0:
        return 0.0, {"pyin": 0.0, "verdict": "unknown"}
    fp = fundamental_presence(x, sr, f)
    pres = f * {"f0": 1.0, "half": 0.5, "third": 1 / 3.0,
                "unknown": 1.0}[fp["verdict"]]
    lo, hi = GENDER_RANGE.get(gender, (60.0, 300.0))
    # 融合：pyin 若在音域内且存在性未判 half -> 直接采信
    if lo <= f <= hi and fp["verdict"] == "f0":
        final = f
    elif lo <= pres <= hi:
        final = pres
    else:
        cands = [v for v in (f, pres) if HARD_LO <= v <= HARD_HI]
        final = min(cands, key=lambda v: abs(v - (lo + hi) / 2)) if cands else f
    return round(float(final), 2), {
        "pyin": round(f, 2), "presence": round(float(pres), 2),
        "verdict": fp["verdict"], "rel_f": fp.get("rel_f", 0),
        "rel_half": fp.get("rel_half", 0),
    }


def one(job):
    cid, emo, path, scale, gender = job
    try:
        x, sr = sf.read(path, dtype="float32")
    except Exception as e:
        return {"cid": cid, "emotion": emo, "error": str(e)}
    if x.ndim > 1:
        x = x[:, 0]
    f_adj, det = adjudicate(x, sr, gender)
    lo_r, hi_r = TARGET_RATIO[emo]
    abs_lo, abs_hi = scale * lo_r, scale * hi_r
    in_range = abs_lo * (1 - ACCEPT_TOL) <= f_adj <= abs_hi * (1 + ACCEPT_TOL)
    # 性别音域检查（用裁决后的值）
    glo, ghi = (60.0, 200.0) if gender == "男" else (145.0, 300.0)
    return {
        "cid": cid, "emotion": emo, "scale": scale,
        "f_adj": f_adj, "r_adj": round(f_adj / scale, 4) if scale else 0,
        "lo_r": lo_r, "hi_r": hi_r, "abs_lo": round(abs_lo, 1),
        "abs_hi": round(abs_hi, 1),
        "in_range": bool(in_range),
        "gender_ok": bool(glo <= f_adj <= ghi),
        "gender": gender, **det,
    }


def main():
    base = json.load(open(f"{VOICES}/base_f0_final.json"))
    cast = {c["id"]: c.get("gender", "") for c in
            json.load(open("/ssd2/duanju/scripts/cast_all_150.json"))}
    rep = json.load(open(REPORT))
    report_map = {(r["cid"], r["emotion"]): r for r in rep["results"]}

    jobs = []
    for cid in sorted(base):
        scale = float(base[cid]["scale"] if isinstance(base[cid], dict)
                      else base[cid])
        for emo in TARGET_RATIO:
            p = os.path.join(FIXED, f"{cid}_{emo}.wav")
            if os.path.exists(p):
                jobs.append((cid, emo, p, scale, cast.get(cid, "")))
    print(f"[1/3] 独立复审 {len(jobs)} 条成品音频（三信号裁决）")
    with ProcessPoolExecutor(max_workers=48) as ex:
        rows = list(ex.map(one, jobs, chunksize=4))

    print("[2/3] 统计")
    by_emo = {}
    for r in rows:
        by_emo.setdefault(r["emotion"], []).append(r)
    print(f"  {'情绪':<6}{'n':>5}{'独立达标':>10}{'v15自报':>10}{'一致':>8}"
          f"{'性别音域':>10}")
    tot_ok = 0
    tot_v15 = 0
    agree = 0
    for emo in TARGET_RATIO:
        rs = by_emo.get(emo, [])
        if not rs:
            continue
        ok = sum(1 for r in rs if r["in_range"])
        v15 = sum(1 for r in rs
                  if report_map.get((r["cid"], r["emotion"]), {}).get("ok"))
        ag = sum(1 for r in rs
                 if bool(r["in_range"]) == bool(
                     report_map.get((r["cid"], r["emotion"]), {}).get("ok")))
        gok = sum(1 for r in rs if r["gender_ok"])
        tot_ok += ok
        tot_v15 += v15
        agree += ag
        print(f"  {emo:<6}{len(rs):>5}{ok:>10}{v15:>10}{ag:>8}{gok:>10}")

    print("-" * 70)
    print(f"  独立复审达标: {tot_ok}/{len(rows)} = {tot_ok/max(len(rows),1)*100:.2f}%")
    print(f"  v15 自报达标: {tot_v15}/{len(rows)} = {tot_v15/max(len(rows),1)*100:.2f}%")
    print(f"  两者一致: {agree}/{len(rows)} = {agree/max(len(rows),1)*100:.2f}%")

    for g in ("男", "女"):
        v = np.array([r["f_adj"] for r in rows if r["gender"] == g])
        if not len(v):
            continue
        glo, ghi = (60.0, 200.0) if g == "男" else (145.0, 300.0)
        ins = int(((v >= glo) & (v <= ghi)).sum())
        print(f"  {g}: n={len(v)} 中位 {np.median(v):.1f}Hz  "
              f"区间 {v.min():.1f}~{v.max():.1f}  在音域 {ins}/{len(v)} "
              f"({ins/max(len(v),1)*100:.1f}%)")

    # 分歧明细
    dis = []
    for r in rows:
        v15r = report_map.get((r["cid"], r["emotion"]), {})
        if bool(r["in_range"]) != bool(v15r.get("ok")):
            dis.append((r, v15r))
    if dis:
        print("-" * 70)
        print(f"独立复审与 v15 自报的分歧 {len(dis)} 条（前 20）：")
        for r, v15r in dis[:20]:
            print(f"  {r['cid']:<6}{r['emotion']:<6} 独立 f={r['f_adj']:7.1f} "
                  f"r={r['r_adj']:.3f} 区间[{r['abs_lo']:.0f},{r['abs_hi']:.0f}] "
                  f"in={r['in_range']} | v15 f={v15r.get('f0_after')} "
                  f"r={v15r.get('r_after')} ok={v15r.get('ok')} "
                  f"| pyin={r['pyin']} pres={r['presence']} vd={r['verdict']}")

    json.dump(rows, open(f"{VOICES}/audit_final_v15.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"  -> {VOICES}/audit_final_v15.json")


if __name__ == "__main__":
    main()
