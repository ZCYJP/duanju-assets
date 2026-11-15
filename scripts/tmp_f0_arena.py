"""估计器竞技场：用同一批 round-trip 对比多个 f0 估计器的八度正确率。

评测台（完全无循环）
--------------------
变调算子精确 => 真 f0(变调后) = k × 真 f0(变调前)，k 已知。
对任意估计器 E：
    ratio = E(变调后) / E(变调前)
若 E 两次锁在同一八度，ratio ≈ k；否则 ratio ≈ k×2^j。
=> 「ratio ≈ k 的比例」= E 的八度正确率，且与 E 自身的绝对校准无关。

候选
----
 A. librosa.pyin            （帧级 YIN，工业实现）
 B. yin_diff（自研 CMND）    （v2 主估计器）
 C. comb（梳谱全局峰）       （v13 用的 wide_measure）
 D. ac（自相关峰值）
 E. 多算法投票

另外测一个关键指标：**同算法在原音/变调音上的比值一致性**
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tmp_f0_truth2 import yin_diff, spectrum, pick_peaks  # noqa: E402

VOICES = "/ssd2/duanju/voices"


# ------------------------------------------------------------- 各估计器
def est_pyin(x, sr):
    try:
        f, _, _ = librosa.pyin(x.astype(np.float32), fmin=40.0, fmax=800.0, sr=sr,
                               frame_length=2048, hop_length=256)
        v = f[np.isfinite(f) & (f > 40) & (f < 800)]
        return float(np.median(v)) if len(v) else 0.0
    except Exception:
        return 0.0


def est_yin(x, sr):
    f, _, _ = yin_diff(x, sr, fmin=40.0, fmax=800.0)
    return f


def est_comb(x, sr):
    """梳谱全局峰（v13 的 wide_measure 等价实现）。"""
    S, freqs = spectrum(x, sr, n_fft=1 << int(np.ceil(np.log2(max(len(x), 8192)))))
    df = float(freqs[1] - freqs[0])
    grid = np.arange(40.0, 800.0 + df, df)
    H = np.maximum((2400.0 / grid).astype(int), 1)
    total = np.zeros(len(grid))
    for h in range(1, int(H.max()) + 1):
        mask = H >= h
        if not np.any(mask):
            continue
        f_h = grid[mask] * h
        pos = f_h / df
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, len(freqs) - 1)
        frac = pos - lo
        amp = S[lo] * (1 - frac) + S[hi] * frac
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


def est_ac(x, sr):
    """自相关峰值（限定 40~800Hz）。"""
    xv = x.astype(np.float64)
    if len(xv) > sr * 8:
        xv = xv[:int(sr * 8)]
    xv = xv - xv.mean()
    n = 1 << int(np.ceil(np.log2(2 * len(xv))))
    A = np.fft.rfft(xv, n)
    R = np.fft.irfft(A * np.conj(A), n)[:len(xv)]
    if R[0] <= 0:
        return 0.0
    R = R / R[0]
    lo = max(2, int(sr / 800.0))
    hi = min(len(R) - 1, int(sr / 40.0))
    if hi <= lo:
        return 0.0
    i = int(np.argmax(R[lo:hi])) + lo
    if lo < i < hi:
        y0, y1, y2 = R[i - 1], R[i], R[i + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
        if abs(d) > 1.0:
            d = 0.0
        return sr / (i + d)
    return sr / i


ESTS = {"pyin": est_pyin, "yin": est_yin, "comb": est_comb, "ac": est_ac}


def vote_est(x, sr):
    """多算法投票：取彼此相差 <=8% 的簇中的中位（按簇大小优先）。"""
    vals = {}
    for name, fn in ESTS.items():
        try:
            vals[name] = fn(x, sr)
        except Exception:
            vals[name] = 0.0
    live = {k: v for k, v in vals.items() if v > 0}
    if not live:
        return 0.0, vals
    clusters = []
    for name, v in live.items():
        placed = False
        for c in clusters:
            if abs(v - c["center"]) / max(c["center"], 1e-6) <= 0.08:
                c["vals"].append(v)
                c["names"].append(name)
                c["center"] = float(np.median(c["vals"]))
                placed = True
                break
        if not placed:
            clusters.append({"center": v, "vals": [v], "names": [name]})
    clusters.sort(key=lambda c: (-len(c["vals"]), c["center"]))
    best = clusters[0]
    return float(np.median(best["vals"])), vals


def pitch_shift(x, sr, factor):
    if abs(factor - 1.0) < 1e-6:
        return x.astype(np.float32)
    n_in = len(x)
    rms_in = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    y = librosa.resample(x.astype(np.float32), orig_sr=sr,
                         target_sr=max(int(sr / factor), 1000))
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    Ds = librosa.phase_vocoder(D, rate=1.0 / factor, hop_length=256, n_fft=1024)
    out = np.asarray(librosa.istft(Ds, hop_length=256, n_fft=1024), dtype=np.float32)
    if len(out) < n_in:
        out = np.pad(out, (0, n_in - len(out)))
    else:
        out = out[:n_in]
    rms_out = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    if rms_out > 1e-8 and rms_in > 1e-8:
        out = out * (rms_in / rms_out)
    pk = float(np.max(np.abs(out)))
    if pk > 0.99:
        out = out * (0.99 / pk)
    return out


def one(job):
    path, factor = job
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    y = pitch_shift(x, sr, factor)
    row = {"path": os.path.basename(path), "factor": factor}
    for name, fn in ESTS.items():
        a = fn(x, sr)
        b = fn(y, sr)
        row[name] = {"a": round(a, 2), "b": round(b, 2),
                     "ratio": round(b / a, 4) if a > 0 and b > 0 else 0.0}
    va, _ = vote_est(x, sr)
    vb, _ = vote_est(y, sr)
    row["vote"] = {"a": round(va, 2), "b": round(vb, 2),
                   "ratio": round(vb / va, 4) if va > 0 and vb > 0 else 0.0}
    return row


def main():
    files = []
    for pat in (f"{VOICES}/samples/*_voice.wav",
                f"{VOICES}/samples_ext/*_voice.wav"):
        files += sorted(glob.glob(pat))
    rng = np.random.default_rng(11)
    idx = rng.choice(len(files), size=min(60, len(files)), replace=False)
    sel = [files[i] for i in sorted(idx)]
    factors = (0.5, 0.70710678, 1.41421356, 2.0)
    jobs = [(p, f) for p in sel for f in factors]
    print(f"[1/2] {len(sel)} 个真实人声 × {len(factors)} 个变调因子 = {len(jobs)} 对")

    rows = []
    with ProcessPoolExecutor(max_workers=48) as ex:
        for r in ex.map(one, jobs, chunksize=2):
            rows.append(r)

    print("[2/2] 竞技场统计")
    print("=" * 80)
    print(f"{'估计器':<10}{'八度正确率':>12}{'误差中位':>10}{'误差p90':>10}{'最大':>10}")
    print("-" * 80)
    summary = {}
    for name in list(ESTS.keys()) + ["vote"]:
        ratios = []
        for r in rows:
            d = r[name]
            if d["ratio"] <= 0:
                continue
            ratios.append(d["ratio"] / r["factor"])
        if not ratios:
            continue
        arr = np.array(ratios)
        err = np.abs(arr - 1.0)
        oct_ok = float((err <= 0.12).mean())
        summary[name] = {"oct_ok": oct_ok, "err_med": float(np.median(err)),
                         "err_p90": float(np.percentile(err, 90)),
                         "err_max": float(err.max()), "n": len(arr)}
        print(f"{name:<10}{oct_ok*100:>11.1f}%{np.median(err)*100:>9.2f}%"
              f"{np.percentile(err,90)*100:>9.2f}%{err.max()*100:>9.2f}%")

    print("-" * 80)
    print("按因子分组（八度正确率）：")
    for f in factors:
        line = f"  k={f:.4f}: "
        for name in list(ESTS.keys()) + ["vote"]:
            sub = [r[name]["ratio"] / f for r in rows
                   if abs(r["factor"] - f) < 1e-6 and r[name]["ratio"] > 0]
            if sub:
                a = np.array(sub)
                line += f"{name}={float((np.abs(a-1)<=0.12).mean())*100:5.1f}%  "
        print(line)

    out = f"{VOICES}/f0_arena.json"
    json.dump({"summary": summary, "rows": rows}, open(out, "w"),
              ensure_ascii=False, indent=1)
    print(f"  -> {out}")

    best = max(summary.items(), key=lambda kv: kv[1]["oct_ok"])
    print(f"  最优估计器: {best[0]}  八度正确率 {best[1]['oct_ok']*100:.1f}%")


if __name__ == "__main__":
    main()
