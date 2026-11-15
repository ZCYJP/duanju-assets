"""绝对八度的最终物理裁决：基频存在性（fundamental presence）。

问题
----
round-trip 只能验证「比值」对不对，无法发现「整体偏一个八度」。
绝对八度必须靠**物理事实**裁决：

  真实人声在**基频 f0 处一定有能量**（声带基频分量）。
  而 2*f0 处虽然是谐波，但它的「基频」f0/2 处**是静音**。
  => 若某候选 f 处有峰、而 f/2 处无峰，则 f 就是真 f0；
     若 f/2 处也有峰，则真 f0 更可能是 f/2。

这不是启发式，而是可证伪的物理判据：
  - 在合成信号上验证（已知真 f0）
  - 在真人声上与「性别音域」交叉验证（独立于 f0 估计器）

三个独立信号裁决 scale：
  S1. pyin 绝对读数（round-trip 八度正确率 95.8%）
  S2. 基频存在性（物理判据）
  S3. 性别音域先验（语义先验，与 DSP 无关）
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
from tmp_f0_truth2 import spectrum  # noqa: E402

VOICES = "/ssd2/duanju/voices"
CAST = "/ssd2/duanju/scripts/cast_all_150.json"

# 性别音域（放宽自然变异）
GENDER_RANGE = {
    "男": (75.0, 200.0),
    "女": (150.0, 285.0),
}


def local_peak_energy(S, freqs, f, rel=0.035):
    """f 附近的局部峰能量（取 ±rel 邻域内最大值），并做抛物线细化。"""
    df = float(freqs[1] - freqs[0])
    i0 = int(f / df)
    w = max(2, int(rel * f / df))
    lo = max(1, i0 - w)
    hi = min(len(S) - 1, i0 + w)
    if hi <= lo:
        return 0.0, 0.0
    j = lo + int(np.argmax(S[lo:hi]))
    if j <= 0 or j >= len(S) - 1:
        return float(S[j]), float(freqs[j])
    y0, y1, y2 = S[j - 1], S[j], S[j + 1]
    den = y0 - 2 * y1 + y2
    d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
    if abs(d) > 1.0:
        d = 0.0
    return float(S[j]), float(freqs[j] + d * df)


def fundamental_presence(x, sr, f, df_min=0.0):
    """基频存在性判定。

    返回 dict：
      e_f    在 f 处的峰能量
      e_half 在 f/2 处的峰能量
      e_third在 f/3 处的峰能量
      ratio_half = e_f / (e_half + eps)
      verdict: "f0" | "half" | "third" | "unknown"
    """
    S, freqs = spectrum(x, sr)
    e_f, pf = local_peak_energy(S, freqs, f)
    e_h, ph = local_peak_energy(S, freqs, f / 2.0)
    e_t, pt = local_peak_energy(S, freqs, f / 3.0)
    if e_f <= 0:
        return {"e_f": 0.0, "e_half": 0.0, "e_third": 0.0,
                "ratio_half": 0.0, "ratio_third": 0.0,
                "verdict": "unknown", "peak_at_f": 0.0}
    # 全局峰做参照
    gmax = float(S[:int(np.searchsorted(freqs, 3000.0))].max())
    rel_f = e_f / max(gmax, 1e-12)
    rel_h = e_h / max(gmax, 1e-12)
    rel_t = e_t / max(gmax, 1e-12)
    # 判据：若 f/2（或 f/3）处的能量达到 f 处的 25% 以上，
    #       说明那里也可能是基频 -> 真 f0 更低
    verdict = "f0"
    if rel_h >= 0.25 * max(rel_f, 1e-9) and rel_h > 0.02:
        verdict = "half"
    if rel_t >= 0.25 * max(rel_f, 1e-9) and rel_t > 0.02 and rel_t > rel_h:
        verdict = "third"
    return {"e_f": round(e_f, 3), "e_half": round(e_h, 3), "e_third": round(e_t, 3),
            "rel_f": round(rel_f, 4), "rel_half": round(rel_h, 4),
            "rel_third": round(rel_t, 4),
            "ratio_half": round(e_f / max(e_h, 1e-12), 2),
            "ratio_third": round(e_f / max(e_t, 1e-12), 2),
            "verdict": verdict}


def find_base(cid):
    for d in ("samples", "samples_ext"):
        g = sorted(glob.glob(f"{VOICES}/{d}/{cid}_*_voice.wav"))
        if g:
            return g[0]
    return ""


def one(c):
    cid = c["id"]
    g = c.get("gender", "")
    p = find_base(cid)
    if not p:
        return {"cid": cid, "gender": g, "f0": 0.0, "fp": None}
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    f = est_pyin(x, sr)
    fp = fundamental_presence(x, sr, f) if f > 0 else None
    return {"cid": cid, "gender": g, "f0": round(f, 2), "fp": fp}


def synth_selftest():
    print("=" * 76)
    print("基频存在性判据：合成信号自检")
    print("=" * 76)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tmp_f0_truth2 import synth
    ok = 0
    n = 0
    for f0 in [70, 90, 110, 130, 160, 190, 220, 260]:
        x, sr = synth(f0=f0)
        # 故意用 2*f0 作为候选，检验能否纠正回 f0
        for cand_mult, label in ((1.0, "f0"), (2.0, "2f0")):
            cand = f0 * cand_mult
            r = fundamental_presence(x, sr, cand)
            pred = cand
            if r["verdict"] == "half":
                pred = cand / 2.0
            elif r["verdict"] == "third":
                pred = cand / 3.0
            err = abs(pred - f0) / f0
            good = err <= 0.10
            n += 1
            ok += int(good)
            print(f"  真值 {f0:5.1f}  候选 {cand:6.1f}({label})  "
                  f"判定 {r['verdict']:<8} 推得 {pred:6.1f}  "
                  f"rel_f={r['rel_f']:.3f} rel_half={r['rel_half']:.3f} "
                  f"{'OK' if good else 'FAIL'}")
    print(f"  -> {ok}/{n} 正确")
    return ok, n


def main():
    print("[A] 基频存在性判据自检")
    synth_selftest()

    cast = json.load(open(CAST))
    print(f"\n[B] 150 个基线音色的三信号裁决")
    with ProcessPoolExecutor(max_workers=48) as ex:
        rows = list(ex.map(one, cast, chunksize=2))

    tally = {}
    for g in ("男", "女"):
        grp = [r for r in rows if r["gender"] == g]
        lo, hi = GENDER_RANGE[g]
        v = np.array([r["f0"] for r in grp])
        inr = int(((v >= lo) & (v <= hi)).sum())
        vd = {}
        for r in grp:
            k = r["fp"]["verdict"] if r["fp"] else "unknown"
            vd[k] = vd.get(k, 0) + 1
        print(f"  {g}: n={len(grp)}  pyin 落音域 {inr}/{len(grp)}  "
              f"中位 {np.median(v):.1f}Hz  存在性判定 {vd}")
        tally[g] = {"n": len(grp), "pyin_in_range": inr, "verdicts": vd,
                    "median": float(np.median(v))}

    print("\n[C] 逐角色裁决表")
    out = {}
    factors = {"f0": 1.0, "half": 0.5, "third": 1.0 / 3.0, "unknown": 1.0}
    disagree = []
    for r in rows:
        cid, g = r["cid"], r["gender"]
        fp = r["fp"] or {}
        verdict = fp.get("verdict", "unknown")
        lo, hi = GENDER_RANGE.get(g, (60, 300))
        f_pyin = r["f0"]
        f_pres = f_pyin * factors.get(verdict, 1.0)
        # 综合：优先基频存在性；若其结论导致超界则退回 pyin
        pick = "presence"
        f_final = f_pres
        if f_final <= 0 or not (lo <= f_final <= hi):
            f_final = f_pyin
            pick = "pyin"
        if not (lo <= f_final <= hi):
            pick = "none"
        out[cid] = {"gender": g, "pyin": f_pyin, "presence": round(f_pres, 2),
                    "verdict": verdict, "final": round(f_final, 2),
                    "pick": pick, "range": [lo, hi],
                    "rel_f": fp.get("rel_f", 0), "rel_half": fp.get("rel_half", 0),
                    "rel_third": fp.get("rel_third", 0)}
        if verdict != "f0":
            disagree.append((cid, g, f_pyin, verdict, round(f_pres, 1), pick))

    json.dump(out, open(f"{VOICES}/base_f0_ground.json", "w"),
              ensure_ascii=False, indent=1)

    print(f"  {'cid':<7}{'性别':<6}{'pyin':>9}{'存在性':>9}{'判定':>10}{'最终':>9}  依据")
    for cid, g, fp, vd, fpres, pick in disagree:
        o = out[cid]
        print(f"  {cid:<7}{g:<6}{fp:>9.1f}{fpres:>9.1f}{vd:>10}{o['final']:>9.1f}  {pick}")

    finals = np.array([out[c]["final"] for c in out])
    for g in ("男", "女"):
        v = np.array([out[c]["final"] for c in out if out[c]["gender"] == g])
        lo, hi = GENDER_RANGE[g]
        print(f"  最终 {g}: 中位 {np.median(v):.1f}  区间 {v.min():.1f}~{v.max():.1f}  "
              f"落音域 {int(((v>=lo)&(v<=hi)).sum())}/{len(v)}")
    print(f"  -> {VOICES}/base_f0_ground.json")


if __name__ == "__main__":
    main()
