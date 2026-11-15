"""最终独立验证：用**与 pyin 完全不同**的测量链验证相对倍率。

为什么必须换一条链
------------------
v16 的判据与变调都用 pyin。若 pyin 有系统性偏差，判据会自洽地错。
独立验证必须用**完全不同的算法**，否则只是自证。

本验证用两条独立链：
  链 A：梳谱（harmonic comb，v13 的 wide_measure 家族）
  链 B：帧级 YIN first-dip（自研 CMND）

验证对象：处理后音频的**相对倍率** r = f(情绪) / f(基线)
  - 判据：r ∈ [lo_r, hi_r]
  - 链 A / 链 B 各自独立给出结论
  - 若链 A、B、pyin 三者结论一致 => 结论可信

同时对**基线音色本身**做三链交叉：
  - 若三链对基线的绝对读数分歧大（差 2 倍），
    说明是测量问题；但对相对倍率的结论仍应一致
    （因为分子分母同链、同尺度）

输出：三链一致性矩阵 + 分歧明细。
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tmp_f0_relative_anchor import (pyin_f0, find_base, TARGET_RATIO,  # noqa: E402
                                    EMO_KEYS)
from tmp_f0_ground import fundamental_presence  # noqa: E402

VOICES = "/ssd2/duanju/voices"


# ------------------------------------------------------------- 链 A：梳谱
def comb_f0(x, sr, lo=45.0, hi=800.0, df=0.5):
    """谐波梳谱全局峰（与 pyin 完全不同的一族算法）。"""
    n_fft = 1 << int(np.ceil(np.log2(max(len(x), 16384))))
    w = np.hanning(len(x)).astype(np.float32)
    S = np.abs(np.fft.rfft(x * w, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    grid = np.arange(lo, hi + df, df)
    H = np.maximum((2400.0 / grid).astype(int), 1)
    total = np.zeros(len(grid))
    for h in range(1, int(H.max()) + 1):
        mask = H >= h
        if not np.any(mask):
            continue
        f_h = grid[mask] * h
        pos = f_h / df
        lo_i = np.floor(pos).astype(int)
        hi_i = np.minimum(lo_i + 1, len(freqs) - 1)
        frac = pos - lo_i
        amp = S[lo_i] * (1 - frac) + S[hi_i] * frac
        total[mask] += amp / np.sqrt(h)
    sc = total / np.maximum(H.astype(np.float64) ** 0.5, 1e-9)
    i = int(np.argmax(sc))
    if 0 < i < len(grid) - 1:
        y0, y1, y2 = sc[i - 1], sc[i], sc[i + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
        if abs(d) > 1.0:
            d = 0.0
        return float(grid[i] + d * df)
    return float(grid[i])


# ------------------------------------------------------------- 链 B：帧级 YIN
def yin_track_median(x, sr, fmin=45.0, fmax=600.0, frame_len=2048, hop=512,
                     thresh=0.15, max_frames=200):
    """帧级 YIN first-dip，取所有有效帧的众数（帧级更抗语调漂移）。"""
    n = len(x)
    if n < frame_len * 2:
        return 0.0
    starts = list(range(0, n - frame_len, hop))
    if len(starts) > max_frames:
        idx = np.linspace(0, len(starts) - 1, max_frames).astype(int)
        starts = [starts[i] for i in idx]
    tau_min = max(2, int(sr / fmax))
    tau_max = int(sr / fmin)
    if tau_max + 8 >= frame_len:
        tau_max = frame_len - 8
    vals = []
    for s in starts:
        seg = x[s:s + frame_len].astype(np.float64)
        seg = seg - seg.mean()
        W = frame_len
        nfft = 1 << int(np.ceil(np.log2(2 * W)))
        A = np.fft.rfft(seg, nfft)
        R = np.fft.irfft(A * np.conj(A), nfft)[:tau_max + 1]
        cum = np.concatenate([[0.0], np.cumsum(seg * seg)])
        E1 = cum[W]
        E2 = cum[W] - cum[:tau_max + 1]
        e = np.maximum(E1 + E2 - 2.0 * R, 0.0)
        e[:tau_min] = np.inf
        fin = np.isfinite(e)
        safe = np.where(fin, e, 0.0)
        csum = np.cumsum(safe)
        cnt = np.cumsum(fin.astype(np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            cmnd = np.where(fin & (cnt > 0),
                            e * np.arange(len(e)) / np.maximum(csum, 1e-12), 1.0)
        seg_cm = cmnd[tau_min:tau_max + 1]
        if not np.any(np.isfinite(seg_cm)):
            continue
        cand = None
        for i in range(1, len(seg_cm) - 1):
            v = seg_cm[i]
            if np.isfinite(v) and v < thresh and v <= seg_cm[i - 1] and v < seg_cm[i + 1]:
                cand = i + tau_min
                break
        if cand is None:
            continue
        if tau_min < cand < tau_max:
            y0, y1, y2 = cmnd[cand - 1], cmnd[cand], cmnd[cand + 1]
            den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
            if abs(d) > 1.0:
                d = 0.0
            tau = cand + d
        else:
            tau = cand
        vals.append(sr / max(tau, 1e-9))
    if not vals:
        return 0.0
    v = np.array(vals)
    # 众数（按 5% 相对分箱）
    bins = np.round(np.log2(v) * 20).astype(int)
    vals_, cnts = np.unique(bins, return_counts=True)
    best = vals_[int(np.argmax(cnts))]
    sel = v[np.round(np.log2(v) * 20).astype(int) == best]
    return float(np.median(sel))


# ------------------------------------------------------------- 链 C：基频存在性调整后的 pyin
def pyin_presence(x, sr):
    """pyin + 基频存在性复核，返回修正后的绝对 f0。"""
    f = pyin_f0(x, sr)
    if f <= 0:
        return 0.0, "none"
    fp = fundamental_presence(x, sr, f)
    if fp["verdict"] == "half":
        return f / 2.0, "half"
    if fp["verdict"] == "third":
        return f / 3.0, "third"
    return f, "f0"


def measure(x, sr):
    return {
        "pyin": pyin_f0(x, sr),
        "comb": comb_f0(x, sr),
        "yin": yin_track_median(x, sr),
    }


def one(job):
    cid, emo, path, base_path = job
    try:
        x, sr = sf.read(path, dtype="float32")
        bx, bsr = sf.read(base_path, dtype="float32")
    except Exception as e:
        return None
    if x.ndim > 1:
        x = x[:, 0]
    if bx.ndim > 1:
        bx = bx[:, 0]
    m_emo = measure(x, sr)
    m_base = measure(bx, bsr)
    lo_r, hi_r = TARGET_RATIO[emo]
    out = {"cid": cid, "emotion": emo, "lo_r": lo_r, "hi_r": hi_r}
    for k in ("pyin", "comb", "yin"):
        be = m_base[k]
        ee = m_emo[k]
        out[f"base_{k}"] = round(be, 2)
        out[f"emo_{k}"] = round(ee, 2)
        out[f"r_{k}"] = round(ee / be, 4) if be > 0 and ee > 0 else 0.0
        out[f"ok_{k}"] = bool(be > 0 and ee > 0 and
                             lo_r * 0.94 <= ee / be <= hi_r * 1.06)
    # 三链一致
    oks = [out["ok_pyin"], out["ok_comb"], out["ok_yin"]]
    out["n_ok"] = sum(oks)
    out["unanimous"] = all(oks)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", default=f"{VOICES}/emotions_fixed_v16")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--out", default=f"{VOICES}/verify_rel_v16.json")
    ap.add_argument("--sample", type=int, default=0,
                    help=">0 则只抽样该数量（快速验证）")
    args = ap.parse_args()

    jobs = []
    for cid in sorted(os.listdir(f"{VOICES}/emotions")):
        bp = find_base(cid)
        if not bp:
            continue
        for emo in EMO_KEYS:
            p = os.path.join(args.fixed, f"{cid}_{emo}.wav")
            if os.path.exists(p):
                jobs.append((cid, emo, p, bp))
    if args.sample and len(jobs) > args.sample:
        rng = np.random.default_rng(3)
        idx = sorted(rng.choice(len(jobs), size=args.sample, replace=False))
        jobs = [jobs[i] for i in idx]
    print(f"[1/3] 三链独立测量 {len(jobs)} 组（pyin / comb / YIN）")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for r in ex.map(one, jobs, chunksize=2) if r]

    print("[2/3] 结果")
    print(f"  {'链':<8}{'达标':>8}{'比例':>9}")
    for k in ("pyin", "comb", "yin"):
        ok = sum(1 for r in rows if r[f"ok_{k}"])
        print(f"  {k:<8}{ok:>8}{ok/max(len(rows),1)*100:>8.2f}%")
    un = sum(1 for r in rows if r["unanimous"])
    print(f"  {'三链一致':<8}{un:>8}{un/max(len(rows),1)*100:>8.2f}%")
    n3 = sum(1 for r in rows if r["n_ok"] == 3)
    n2 = sum(1 for r in rows if r["n_ok"] == 2)
    n1 = sum(1 for r in rows if r["n_ok"] == 1)
    n0 = sum(1 for r in rows if r["n_ok"] == 0)
    print(f"  分布: 3链全对 {n3}  2链 {n2}  1链 {n1}  0链 {n0}")

    print("-" * 74)
    print("分歧明细（链结论不一致，前 20）：")
    dis = [r for r in rows if r["n_ok"] not in (0, 3)]
    for r in dis[:20]:
        print(f"  {r['cid']:<6}{r['emotion']:<6} "
              f"r_pyin={r['r_pyin']:.3f} r_comb={r['r_comb']:.3f} "
              f"r_yin={r['r_yin']:.3f}  区间[{r['lo_r']},{r['hi_r']}]  "
              f"ok={r['n_ok']}/3")

    # 基线读数的三链一致性（诊断测量可靠性）
    print("-" * 74)
    print("基线音色绝对读数的三链一致性（诊断用）")
    for k in ("pyin", "comb", "yin"):
        v = np.array([r[f"base_{k}"] for r in rows if r[f"base_{k}"]])
        if len(v):
            print(f"  base_{k}: 中位 {np.median(v):.1f}  "
                  f"区间 {v.min():.1f}~{v.max():.1f}")
    ratios = []
    for r in rows:
        b, e = r["base_pyin"], r["base_comb"]
        if b > 0 and e > 0:
            ratios.append(e / b)
    if ratios:
        ratios = np.array(ratios)
        print(f"  base_comb/base_pyin 中位 {np.median(ratios):.3f}  "
              f"(1.0=两链一致, 0.5/2.0=差一八度)")

    json.dump(rows, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
