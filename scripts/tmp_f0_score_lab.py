"""梳谱打分变体评测（诚实 roundtrip：真实人声 + 已知变调倍率）。

问题
----
尺度锚定法在真实人声上的残余失败，机制清晰：
  开环变调后音频已落在正确位置，但梳谱得分偶尔把 2*f0 判成 f0，
  精修循环据此把已经正确的音频继续调坏。
=> 必须把「真 f0 vs 2*f0」的判别做可靠。

协议
----
取真实基线音频 -> 变调已知倍率 fac -> 用尺度锚定法测回
真值 = 原音频真 f0 * fac；尺度 = 原音频的梳谱宽测值。
此协议与生产用法完全一致（尺度给的是宽测值，不是真值）。

变体
----
V1  候选枚举 + 梳谱得分（当前 v12 实现）
V2  V1 + 次谐波惩罚（S(f/2) 强则降八度）
V3  自相关首峰播种 + 梳谱细化
V4  V3 + 次谐波惩罚
V5  V4 + 时域周期性验证
V6  V4 + 谐波覆盖度（odd harmonics 缺失检测）
"""
import glob
import json
import os

import librosa
import numpy as np
import soundfile as sf

VOICES = "/ssd2/duanju/voices"
COMB_MAX = 2400.0
NORM_POW = 0.5
FMAX = 1200.0


# ------------------------------------------------------------- 基础
def spec(x, sr):
    n_fft = 1 << int(np.ceil(np.log2(max(len(x), 4096))))
    w = np.hanning(len(x)).astype(np.float32)
    S = np.abs(np.fft.rfft(x * w, n=n_fft))
    return S, np.fft.rfftfreq(n_fft, 1.0 / sr)


def amp_at(S, freqs, f):
    if f <= 0 or f > freqs[-1]:
        return 0.0
    df = freqs[1] - freqs[0]
    pos = f / df
    i = int(pos)
    if i >= len(freqs) - 1:
        return float(S[-1])
    frac = pos - i
    return float(S[i] * (1 - frac) + S[i + 1] * frac)


def comb_score_at(S, freqs, f, comb_max=COMB_MAX, norm_pow=NORM_POW):
    if f <= 0:
        return 0.0
    H = max(int(comb_max / f), 1)
    total = 0.0
    for h in range(1, H + 1):
        fh = f * h
        if fh > freqs[-1]:
            break
        total += amp_at(S, freqs, fh) / np.sqrt(h)
    return total / (H ** norm_pow)


def comb_curve(S, freqs, grid, comb_max=COMB_MAX, norm_pow=NORM_POW):
    H = np.maximum((comb_max / grid).astype(int), 1)
    total = np.zeros(len(grid))
    for h in range(1, int(H.max()) + 1):
        mask = H >= h
        if not np.any(mask):
            continue
        f_h = grid[mask] * h
        df = freqs[1] - freqs[0]
        pos = f_h / df
        lo_i = np.floor(pos).astype(int)
        hi_i = np.minimum(lo_i + 1, len(freqs) - 1)
        frac = pos - lo_i
        amp = S[lo_i] * (1 - frac) + S[hi_i] * frac
        total[mask] += amp / np.sqrt(h)
    return total / np.maximum(H.astype(np.float64) ** norm_pow, 1e-9)


def acf_first_peak(x, sr, fmin=45.0, fmax=800.0, frame=8192):
    """自相关：首个达到全局峰值 60% 的显著峰 -> 周期候选。"""
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    seg = x[:frame].astype(np.float64)
    seg = seg - seg.mean()
    n = len(seg)
    acf = np.correlate(seg, seg, "full")[n - 1:]
    e = acf[0]
    if e <= 0:
        return []
    a = acf / e
    lo = max(2, int(sr / fmax))
    hi = min(n - 2, int(sr / fmin))
    if hi <= lo + 2:
        return []
    seg_a = a[lo:hi]
    gmax = float(np.max(seg_a))
    if gmax <= 0:
        return []
    out = []
    thr = 0.60 * gmax
    for i in range(1, len(seg_a) - 1):
        if seg_a[i] >= seg_a[i - 1] and seg_a[i] >= seg_a[i + 1] \
                and seg_a[i] >= thr:
            tau = lo + i
            if 2 <= tau < n - 1:
                y0, y1, y2 = seg_a[i - 1], seg_a[i], seg_a[i + 1]
                den = y0 - 2 * y1 + y2
                d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
                if abs(d) > 1.0:
                    d = 0.0
                out.append(sr / (tau + d))
            break
    return out


def periodicity(x, sr, f0, frame=8192):
    """候选 f0 处的归一化自相关（周期性强度，0~1）。"""
    if f0 <= 0:
        return 0.0
    tau = sr / f0
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    seg = x[:frame].astype(np.float64)
    seg = seg - seg.mean()
    n = len(seg)
    t = int(round(tau))
    if t < 2 or t >= n - 1:
        return 0.0
    acf = np.correlate(seg, seg, "full")[n - 1:]
    if acf[0] <= 0:
        return 0.0
    a = acf[t] / acf[0]
    b = 0.5 * (acf[t - 1] + acf[t + 1]) / acf[0]
    return float(a - b)


def harmonic_coverage(S, freqs, f, comb_max=COMB_MAX):
    """谐波覆盖度：真 f0 的所有谐波都该有能量；
    2*f0 的谐波集只有偶数倍，odd 缺失。
    返回 (有能量谐波占比, 能量中位数)
    """
    H = max(int(comb_max / f), 1)
    if H < 3:
        return 0.0, 0.0
    amps = []
    for h in range(1, H + 1):
        fh = f * h
        if fh > freqs[-1]:
            break
        amps.append(amp_at(S, freqs, fh))
    if not amps:
        return 0.0, 0.0
    med = float(np.median(amps)) + 1e-12
    cover = float(np.mean(np.array(amps) > med * 0.35))
    return cover, med


# ------------------------------------------------------------- 变体
def _candidates(S, freqs, f_raw, lo, hi, ks=range(1, 13), pyin_f=0.0):
    cands = set()
    for k in ks:
        c = f_raw / k
        if lo <= c <= hi:
            cands.add(round(c, 3))
    if pyin_f > 0:
        for k in range(0, 6):
            c = pyin_f / (2 ** k)
            if lo <= c <= hi:
                cands.add(round(c, 3))
    if not cands:
        cands = {round(f_raw, 3)}
    return sorted(cands)


def _refine(S, freqs, c, lo, hi, df=0.5):
    g = np.arange(max(lo, c * 0.96), min(hi, c * 1.04) + df * 0.5, df * 0.25)
    if len(g) < 5:
        return c
    sc = comb_curve(S, freqs, g)
    j = int(np.argmax(sc))
    if 0 < j < len(g) - 1:
        y0, y1, y2 = sc[j - 1], sc[j], sc[j + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
        if abs(d) > 1.0:
            d = 0.0
        return float(g[j] + d * (g[1] - g[0]))
    return float(g[j])


def v1(x, sr, scale, S, freqs, f_raw, pyin_f, lo, hi):
    cands = _candidates(S, freqs, f_raw, lo, hi, pyin_f=pyin_f)
    scored = [(c, comb_score_at(S, freqs, c)) for c in cands]
    best = max(scored, key=lambda t: t[1])
    return _refine(S, freqs, best[0], lo, hi)


def v2(x, sr, scale, S, freqs, f_raw, pyin_f, lo, hi):
    """V1 + 次谐波惩罚：若 f/2 处能量强（相对 f 的谐波中位数），降八度。"""
    cands = _candidates(S, freqs, f_raw, lo, hi, pyin_f=pyin_f)
    scored = []
    for c in cands:
        s = comb_score_at(S, freqs, c)
        _, med = harmonic_coverage(S, freqs, c)
        if med > 0:
            sub = amp_at(S, freqs, c / 2.0)
            if sub > med * 0.55:
                s *= 0.45
        scored.append((c, s))
    best = max(scored, key=lambda t: t[1])
    return _refine(S, freqs, best[0], lo, hi)


def v3(x, sr, scale, S, freqs, f_raw, pyin_f, lo, hi):
    """自相关首峰播种 + 梳谱细化。"""
    cands = set()
    for f in acf_first_peak(x, sr):
        for k in (1, 2, 3):
            c = f / k
            if lo <= c <= hi:
                cands.add(round(c, 3))
    for k in range(1, 13):
        c = f_raw / k
        if lo <= c <= hi:
            cands.add(round(c, 3))
    if not cands:
        cands = {round(f_raw, 3)}
    scored = [(c, comb_score_at(S, freqs, c)) for c in sorted(cands)]
    best = max(scored, key=lambda t: t[1])
    return _refine(S, freqs, best[0], lo, hi)


def v4(x, sr, scale, S, freqs, f_raw, pyin_f, lo, hi):
    """V3 + 次谐波惩罚。"""
    cands = set()
    for f in acf_first_peak(x, sr):
        for k in (1, 2, 3):
            c = f / k
            if lo <= c <= hi:
                cands.add(round(c, 3))
    for k in range(1, 13):
        c = f_raw / k
        if lo <= c <= hi:
            cands.add(round(c, 3))
    if not cands:
        cands = {round(f_raw, 3)}
    scored = []
    for c in sorted(cands):
        s = comb_score_at(S, freqs, c)
        _, med = harmonic_coverage(S, freqs, c)
        if med > 0:
            sub = amp_at(S, freqs, c / 2.0)
            if sub > med * 0.55:
                s *= 0.45
        scored.append((c, s))
    best = max(scored, key=lambda t: t[1])
    return _refine(S, freqs, best[0], lo, hi)


def v5(x, sr, scale, S, freqs, f_raw, pyin_f, lo, hi):
    """V4 + 时域周期性加权。"""
    cands = set()
    for f in acf_first_peak(x, sr):
        for k in (1, 2, 3):
            c = f / k
            if lo <= c <= hi:
                cands.add(round(c, 3))
    for k in range(1, 13):
        c = f_raw / k
        if lo <= c <= hi:
            cands.add(round(c, 3))
    if not cands:
        cands = {round(f_raw, 3)}
    scored = []
    for c in sorted(cands):
        s = comb_score_at(S, freqs, c)
        _, med = harmonic_coverage(S, freqs, c)
        if med > 0:
            sub = amp_at(S, freqs, c / 2.0)
            if sub > med * 0.55:
                s *= 0.45
        p = periodicity(x, sr, c)
        scored.append((c, s * (0.55 + 0.9 * p)))
    best = max(scored, key=lambda t: t[1])
    return _refine(S, freqs, best[0], lo, hi)


def v6(x, sr, scale, S, freqs, f_raw, pyin_f, lo, hi):
    """V4 + 谐波覆盖度加权。"""
    cands = set()
    for f in acf_first_peak(x, sr):
        for k in (1, 2, 3):
            c = f / k
            if lo <= c <= hi:
                cands.add(round(c, 3))
    for k in range(1, 13):
        c = f_raw / k
        if lo <= c <= hi:
            cands.add(round(c, 3))
    if not cands:
        cands = {round(f_raw, 3)}
    scored = []
    for c in sorted(cands):
        s = comb_score_at(S, freqs, c)
        cov, med = harmonic_coverage(S, freqs, c)
        if med > 0:
            sub = amp_at(S, freqs, c / 2.0)
            if sub > med * 0.55:
                s *= 0.45
        scored.append((c, s * (0.5 + cov)))
    best = max(scored, key=lambda t: t[1])
    return _refine(S, freqs, best[0], lo, hi)


VARIANTS = {"V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5, "V6": v6}


# ------------------------------------------------------------- 变调
def shift(x, sr, factor):
    n_in = len(x)
    rms_in = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    y = librosa.resample(x.astype(np.float32), orig_sr=sr,
                         target_sr=max(int(sr / factor), 1000))
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    Ds = librosa.phase_vocoder(D, rate=1.0 / factor, hop_length=256, n_fft=1024)
    out = np.asarray(librosa.istft(Ds, hop_length=256, n_fft=1024), dtype=np.float32)
    if len(out) < n_in:
        out = np.pad(out, (0, n_in - len(out)))
    elif len(out) > n_in:
        out = out[:n_in]
    rms_out = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    if rms_out > 1e-8 and rms_in > 1e-8:
        out = out * (rms_in / rms_out)
    pk = float(np.max(np.abs(out)))
    if pk > 0.99:
        out = out * (0.99 / pk)
    return out


def pyin_f0(x, sr):
    try:
        f, _, _ = librosa.pyin(x.astype(np.float32), fmin=45, fmax=800, sr=sr,
                               frame_length=2048, hop_length=512)
        v = f[(f > 45) & (f < 800)]
        return float(np.median(v)) if len(v) else 0.0
    except Exception:
        return 0.0


# ------------------------------------------------------------- 评测
def run():
    print("=" * 96)
    print("梳谱打分变体评测（诚实 roundtrip）")
    print("=" * 96)
    files = sorted(glob.glob(f"{VOICES}/samples/*_voice.wav")) + \
            sorted(glob.glob(f"{VOICES}/samples_ext/*_voice.wav"))
    step = max(1, len(files) // 40)
    files = files[::step][:40]
    facs = (0.5, 0.7, 0.85, 1.0, 1.3, 1.6, 2.0)
    errs = {k: [] for k in VARIANTS}
    bad = {k: [] for k in VARIANTS}

    for p in files:
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        S0, f0fr = spec(x, sr)
        g0 = np.arange(45, 800.5, 0.5)
        scale = float(g0[int(np.argmax(comb_curve(S0, f0fr, g0)))])
        if scale <= 0:
            continue
        # 真值用「该文件在宽测下的梳谱峰」作参考（与生产口径一致）
        f0_ref = scale
        for fac in facs:
            y = x if abs(fac - 1.0) < 1e-6 else shift(x, sr, fac)
            truth = f0_ref * fac
            S, fr = spec(y, sr)
            lo = max(35.0, scale * 0.50)
            hi = min(1200.0, scale * 2.00)
            g = np.arange(lo, hi + 0.5, 0.5)
            sc = comb_curve(S, fr, g)
            i = int(np.argmax(sc))
            f_raw = float(g[i])
            pf = pyin_f0(y, sr)
            for name, fn in VARIANTS.items():
                est = fn(y, sr, scale, S, fr, f_raw, pf, lo, hi)
                if est > 0:
                    e = abs(est / truth - 1) * 100
                    errs[name].append(e)
                    if e > 15:
                        bad[name].append((os.path.basename(p)[:10], fac,
                                          round(truth, 1), round(est, 1),
                                          round(f_raw, 1)))
                else:
                    errs[name].append(200.0)

    print("%-6s %8s %8s %8s %8s %10s" %
          ("变体", "中位", "p75", "p90", "最大", "八度正确率"))
    for name in VARIANTS:
        a = np.array(errs[name])
        print("%-6s %7.2f%% %7.2f%% %7.2f%% %7.2f%% %9.1f%%  (n=%d)"
              % (name, np.median(a), np.percentile(a, 75),
                 np.percentile(a, 90), np.max(a),
                 np.mean(a < 15) * 100, len(a)))
    best_name = min(VARIANTS, key=lambda n: np.percentile(errs[n], 90))
    print(f"\n最优(p90): {best_name}")
    print("  失败样例:")
    for r in bad[best_name][:12]:
        print("    %-12s x%.2f 真值=%7.1f 读数=%7.1f raw=%7.1f" % r)


if __name__ == "__main__":
    run()
