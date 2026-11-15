"""独立 f0 真值估计器 v2：YIN(CMND) 为主 + 谐波阶梯 + 倒谱 交叉验证。

设计目标
--------
为「9 情绪音高回正」提供一个**不循环论证**的绝对 f0 测量。

为什么之前的判据不可信
----------------------
v13/v14 用「窄窗读数」当达标判据，而窄窗以 target 为中心（±15%）。
于是「只要窗内能找到峰，r_after 就自动落进达标区间」= 假达标。
真正的风险在 scale（基线尺度）的**绝对八度**：错 2 倍 => 性别漂移。

本模块的三估计器
----------------
1. **YIN（累积均值归一化差分）** —— 主估计器。
   抗倍频关键：d'(tau) 取**第一个**跌破阈值的局部极小点，
   而非全局最小（后者常落在 tau0 的整数倍上）。
2. **谐波阶梯（绝对容差）** —— 辅助/交叉验证。
   绝对容差 tol = max(2.5*df, 0.015*d)，与 FFT 分辨率绑定，
   不随谐波阶数放大（这是 v1 失败的原因）。
3. **倒谱** —— 辅助。quefrency 峰值给出周期候选。

三者投票（相差 <=6% 视为同一声部），取多数簇的中心。
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

VOICES = "/ssd2/duanju/voices"


# ------------------------------------------------------------- 预处理
def _trim_voiced(x, sr, top_db=35.0):
    """能量裁剪：保留有声段，去掉首尾静音。"""
    if len(x) < sr // 10:
        return x
    hop = max(256, int(sr * 0.01))
    n = len(x) // hop
    if n < 4:
        return x
    fr = x[:n * hop].reshape(n, hop)
    e = np.sqrt(np.mean(fr.astype(np.float64) ** 2, axis=1)) + 1e-12
    thr = e.max() * (10 ** (-top_db / 20.0))
    idx = np.where(e > thr)[0]
    if len(idx) == 0:
        return x
    a = max(0, idx[0] * hop - hop)
    b = min(len(x), (idx[-1] + 1) * hop + hop)
    return x[a:b]


def spectrum(x, sr, n_fft=None):
    if n_fft is None:
        n_fft = 1 << int(np.ceil(np.log2(max(len(x), 16384))))
    w = np.hanning(len(x)).astype(np.float32)
    X = np.abs(np.fft.rfft(x * w, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    k = np.hanning(7)
    k /= k.sum()
    X = np.convolve(X, k, mode="same")
    return X, freqs


# ------------------------------------------------------------- YIN
def yin_diff(x, sr, fmin=45.0, fmax=600.0, thresh=0.15):
    """YIN 累积均值归一化差分函数。

    d(tau) = sum_{j} (x[j] - x[j+tau])^2
           = E1 + E2(tau) - 2*R(tau)
    R(tau) 用 FFT 自相关一次算出。

    返回 (f0, tau, cmnd)
    """
    xv = _trim_voiced(x, sr)
    tau_min = max(2, int(sr / fmax))
    tau_max = int(sr / fmin)
    W = len(xv)
    if W < tau_max + 256:
        return 0.0, 0.0, 1.0
    seg = xv.astype(np.float64)
    nfft = 1 << int(np.ceil(np.log2(2 * W)))
    A = np.fft.rfft(seg, nfft)
    R = np.fft.irfft(A * np.conj(A), nfft)[:tau_max + 1]
    cum = np.concatenate([[0.0], np.cumsum(seg * seg)])
    E1 = cum[W]                      # sum x[0..W-1]^2
    E2 = cum[W] - cum[:tau_max + 1]  # sum x[tau..W-1]^2
    e = np.maximum(E1 + E2 - 2.0 * R, 0.0)
    e[:tau_min] = np.inf
    fin = np.isfinite(e)
    safe = np.where(fin, e, 0.0)
    csum = np.cumsum(safe)
    cnt = np.cumsum(fin.astype(np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        cmnd = np.where(fin & (cnt > 0),
                        e * np.arange(len(e)) / np.maximum(csum, 1e-12),
                        1.0)
    seg_cm = cmnd[tau_min:tau_max + 1]
    if not np.any(np.isfinite(seg_cm)):
        return 0.0, 0.0, 1.0
    cand = None
    for i in range(1, len(seg_cm) - 1):
        v = seg_cm[i]
        if np.isfinite(v) and v < thresh and v <= seg_cm[i - 1] and v < seg_cm[i + 1]:
            cand = i + tau_min
            break
    if cand is None:
        cand = int(np.nanargmin(seg_cm)) + tau_min
    if tau_min < cand < tau_max:
        y0, y1, y2 = cmnd[cand - 1], cmnd[cand], cmnd[cand + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
        if abs(d) > 1.0:
            d = 0.0
        tau = cand + d
    else:
        tau = cand
    return sr / max(tau, 1e-9), float(tau), float(cmnd[cand])


# ------------------------------------------------------------- 谐波阶梯
def pick_peaks(S, freqs, fmax=3000.0, top=32, rel=0.05):
    lim = int(np.searchsorted(freqs, fmax))
    S2 = S[:lim].copy()
    if len(S2) < 5:
        return []
    gmax = float(S2.max())
    if gmax <= 0:
        return []
    out = [i for i in range(2, len(S2) - 2)
           if S2[i] >= S2[i - 1] and S2[i] > S2[i + 1] and S2[i] >= gmax * rel]
    out.sort(key=lambda i: S2[i], reverse=True)
    out = sorted(out[:top])
    df = freqs[1] - freqs[0]
    peaks = []
    for i in out:
        y0, y1, y2 = S2[i - 1], S2[i], S2[i + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
        if abs(d) > 1.0:
            d = 0.0
        peaks.append((float(freqs[i] + d * df), float(S2[i])))
    return peaks


def ladder_candidates(peaks, f_lo=55.0, f_hi=500.0):
    fs = [p[0] for p in peaks if p[0] >= f_lo * 0.75]
    cands = set()
    for i in range(len(fs)):
        for j in range(i + 1, len(fs)):
            d = fs[j] - fs[i]
            if d <= 1e-6:
                continue
            for m in (1, 2, 3):
                for k in (1, 2, 3):
                    v = d * m / k
                    if f_lo * 0.7 <= v <= f_hi * 1.3:
                        cands.add(round(v, 3))
    return sorted(cands)


def ladder_score(peaks, d, df):
    """绝对容差匹配。返回 (覆盖率, 命中数, 1阶能量占比, 平均残差Hz)"""
    if not peaks:
        return 0.0, 0, 0.0, 1e9
    tol = max(2.5 * df, 0.015 * d)
    tot_w = hit_w = h1_w = 0.0
    res_sum = 0.0
    hit = 0
    for pf, pa in peaks:
        tot_w += pa
        k = max(1, int(round(pf / d)))
        r = pf - k * d
        if abs(r) <= tol:
            hit += 1
            hit_w += pa
            res_sum += abs(r)
            if k == 1:
                h1_w += pa
    cov = hit_w / max(tot_w, 1e-9)
    return cov, hit, h1_w / max(tot_w, 1e-9), (res_sum / hit if hit else 1e9)


def best_ladder(peaks, df, f_lo=55.0, f_hi=500.0):
    cands = ladder_candidates(peaks, f_lo=f_lo, f_hi=f_hi)
    if not cands:
        return 0.0, 0.0, 0, 0.0, 0.0
    rows = []
    for d in cands:
        cov, hit, h1, res = ladder_score(peaks, d, df)
        rows.append((d, cov, hit, h1, res))
    mx = max(r[1] for r in rows)
    good = [r for r in rows if r[1] >= mx * 0.98]
    # f0 是**最大的**公共步长；并列时取残差更小者
    good.sort(key=lambda r: (-r[0], r[4]))
    d, cov, hit, h1, res = good[0]
    return d, cov, hit, h1, res


# ------------------------------------------------------------- 倒谱
def cepstrum_f0(x, sr, fmin=45.0, fmax=600.0, top=6):
    n_fft = 1 << int(np.ceil(np.log2(max(len(x), 16384))))
    w = np.hanning(len(x)).astype(np.float32)
    X = np.fft.rfft(x * w, n=n_fft)
    logm = np.log(np.maximum(np.abs(X), 1e-10))
    ceps = np.fft.irfft(logm, n=n_fft)
    lo = max(1, int(sr / fmax))
    hi = min(len(ceps) - 1, int(sr / fmin))
    if hi <= lo + 2:
        return []
    seg = np.abs(ceps[lo:hi])
    if seg.max() <= 0:
        return []
    out = []
    for _ in range(top):
        i = int(np.argmax(seg))
        if seg[i] <= 0:
            break
        if 0 < i < len(seg) - 1:
            y0, y1, y2 = seg[i - 1], seg[i], seg[i + 1]
            den = y0 - 2 * y1 + y2
            dd = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
            if abs(dd) > 1.0:
                dd = 0.0
            tq = (lo + i + dd) / sr
        else:
            tq = (lo + i) / sr
        out.append((1.0 / tq if tq > 0 else 0.0, float(seg[i])))
        w2 = max(3, int(0.004 * sr))
        seg[max(0, i - w2):min(len(seg), i + w2 + 1)] = 0
    out.sort(key=lambda z: z[0])
    return out


# ------------------------------------------------------------- 综合
def measure_truth(x, sr, fmin=45.0, fmax=600.0, top=32, rel=0.05):
    """独立测量绝对 f0（三估计器投票）。"""
    S, freqs = spectrum(x, sr)
    df = float(freqs[1] - freqs[0])
    peaks = pick_peaks(S, freqs, top=top, rel=rel)
    res = {"n_peaks": len(peaks)}

    f_yin, tau_yin, cmnd = yin_diff(x, sr, fmin=fmin, fmax=fmax)
    res["yin"] = round(f_yin, 2)
    res["yin_cmnd"] = round(cmnd, 4)
    res["yin_tau"] = round(tau_yin, 2)

    d, cov, hit, h1, resd = best_ladder(peaks, df, f_lo=fmin, f_hi=fmax)
    res["ladder_d"] = round(d, 2)
    res["ladder_cov"] = round(cov, 4)
    res["ladder_h1"] = round(h1, 4)
    res["ladder_hit"] = hit

    ceps = cepstrum_f0(x, sr, fmin=fmin, fmax=fmax)
    res["cep"] = [[round(float(f), 2), round(float(p), 2)] for f, p in ceps[:4]]

    # 投票：YIN 主票 + 阶梯 + 倒谱最低峰
    votes = []
    if f_yin > 0:
        votes.append(("yin", f_yin))
    if d > 0:
        votes.append(("lad", d))
    if ceps:
        votes.append(("cep", ceps[0][0]))

    clusters = []
    for name, v in votes:
        placed = False
        for c in clusters:
            if abs(v - c["center"]) / max(c["center"], 1e-6) <= 0.06:
                c["vals"].append(v)
                c["names"].append(name)
                c["center"] = float(np.mean(c["vals"]))
                placed = True
                break
        if not placed:
            clusters.append({"center": v, "vals": [v], "names": [name]})
    if not clusters:
        res["f0"] = 0.0
        return res
    # 优先含 yin 的簇，否则最大簇
    yin_cl = [c for c in clusters if "yin" in c["names"]]
    pick = yin_cl[0] if yin_cl else max(clusters, key=lambda c: len(c["vals"]))
    res["f0"] = round(float(pick["center"]), 2)
    res["n_vote"] = len(pick["vals"])
    res["n_cluster"] = len(clusters)
    res["agree"] = pick["names"]
    res["clusters"] = [[round(c["center"], 2), c["names"]] for c in clusters]
    return res


def measure_file(path, **kw):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    return measure_truth(x, sr, **kw)


# ------------------------------------------------------------- 合成自检
def synth(sr=24000, dur=2.0, f0=120.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * dur)) / sr
    f = f0 * (1 + 0.01 * np.sin(2 * np.pi * 4.5 * t))
    ph = 2 * np.pi * np.cumsum(f) / sr
    x = np.zeros_like(t)
    for h in range(1, 40):
        if h * f0 > 5000:
            break
        x += np.sin(h * ph) / (h ** 1.1)
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), 1 / sr)
    G = np.ones_like(fr)
    for fc, bw, g in [(500, 90, 12), (1400, 120, 8), (2500, 160, 5), (3500, 200, 3)]:
        G *= (1 + g * np.exp(-((fr - fc) ** 2) / (2 * bw ** 2)))
    x = np.fft.irfft(X * G, len(x)).astype(np.float32)
    x += rng.normal(0, 0.002, len(x)).astype(np.float32)
    return x.astype(np.float32), sr


def selftest():
    print("=" * 78)
    print("真值估计器 v2（YIN主 + 阶梯 + 倒谱）：合成信号绝对精度")
    print("=" * 78)
    errs = []
    for f0 in [65, 80, 100, 120, 150, 180, 220, 260, 320, 400]:
        x, sr = synth(f0=f0)
        r = measure_truth(x, sr)
        e = abs(r["f0"] - f0) / f0
        errs.append(e)
        print(f"  真值 {f0:5.1f} -> 测得 {r['f0']:7.2f}  误差 {e*100:6.2f}%  "
              f"yin={r['yin']:.1f}(cmnd={r['yin_cmnd']:.2f})  "
              f"lad={r['ladder_d']:.1f}  cep={[c[0] for c in r['cep'][:2]]}  "
              f"agree={r.get('agree')}")
    errs = np.array(errs)
    print("-" * 78)
    print(f"中位 {np.median(errs)*100:.3f}%  p90 {np.percentile(errs,90)*100:.3f}%  "
          f"最大 {errs.max()*100:.3f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--file", default="")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.file:
        print(json.dumps(measure_file(args.file), ensure_ascii=False, indent=2))
