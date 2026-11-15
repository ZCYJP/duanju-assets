"""f0 测量流程评测台 v2（黄金标准：合成真值 + 真实一致性）。

核心流程（候选枚举 + 窄带精测 + 突出度择一）
---------------------------------------------
1. 宽测：梳状谱在 45~800Hz 连续搜峰 -> f_raw（可能有倍频错）
2. 候选枚举：f_raw, f_raw/2, f_raw/3, f_raw/4, f_raw/6, f_raw/8
3. 窄带精测：每个候选在 [c*0.85, c*1.20] 内细扫梳状谱
4. 突出度：曲线峰值 / 中位数（真 f0 处谐波全对齐，峰最尖）
5. 择一：在满足合理性约束的候选中，选突出度最高者

合成器：频域共振峰构造（绝对稳定，不会溢出）
"""
import glob
import json
import os

import librosa
import numpy as np
import soundfile as sf

VOICES = "/ssd2/duanju/voices"


# ================================================================== 合成器
def synth_voice(f0, sr=24000, dur=2.5, seed=0,
                formants=((700, 90), (1220, 110), (2600, 160), (3400, 200)),
                f0_drift=0.03, jitter=0.012):
    """源-滤波器合成语音（频域共振峰，稳定无溢出）。"""
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    t = np.arange(n) / sr
    vib = 1.0 + f0_drift * np.sin(2 * np.pi * 4.7 * t + rng.uniform(0, 6.28))
    # 声门脉冲串（频域构造：谐波衰减 + 相位抖动）
    N = 1 << int(np.ceil(np.log2(n * 2)))
    X = np.zeros(N // 2 + 1, dtype=complex)
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    phase0 = rng.uniform(0, 2 * np.pi, size=40)
    h = 1
    while h * f0 < sr / 2 - 100:
        # 该谐波的平均幅度（随 h 衰减）
        amp = (1.0 / h ** 1.2) * (1.0 + 0.5 * rng.uniform(-0.3, 0.3))
        # 颤音导致的频率展宽：用一个小的频率抖动量近似
        df = f0_drift * f0 * 0.5
        k = int(round(h * f0 / (sr / N)))
        lo = max(0, k - int(df / (sr / N)))
        hi = min(len(X) - 1, k + int(df / (sr / N)) + 1)
        for kk in range(lo, hi + 1):
            w = np.exp(-((freqs[kk] - h * f0) ** 2) / (2 * df ** 2 + 1e-9))
            X[kk] += amp * w * np.exp(1j * phase0[h % 40])
        h += 1
    # 共振峰包络（频域增益）
    G = np.ones_like(freqs)
    for fc, bw in formants:
        G += 12.0 / (1.0 + ((freqs - fc) / (bw * 1.6)) ** 2)
    X = X * G
    X += 0.02 * (rng.standard_normal(len(X)) +
                 1j * rng.standard_normal(len(X))) * G * 0.02
    x = np.fft.irfft(X, n=N)[:n].real
    # 声门包络（音节级起伏）
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 2.3 * t + 0.7)
    env = np.clip(env, 0.05, 1.0)
    x = x * env
    pk = np.max(np.abs(x))
    if pk > 0 and np.isfinite(pk):
        x = x / pk * 0.55
    return np.nan_to_num(x).astype(np.float32)


# ================================================================== 梳状谱
def spec(x, sr):
    n_fft = 1 << int(np.ceil(np.log2(max(len(x), 4096))))
    w = np.hanning(len(x)).astype(np.float32)
    X = np.abs(np.fft.rfft(x * w, n=n_fft))
    return X, np.fft.rfftfreq(n_fft, 1.0 / sr)


def comb_curve(S, freqs, grid, comb_max=2400.0, weight="inv_sqrt",
               norm_pow=0.5):
    H = np.maximum((comb_max / grid).astype(int), 1)
    hmax = int(H.max())
    total = np.zeros(len(grid))
    for h in range(1, hmax + 1):
        mask = H >= h
        if not np.any(mask):
            continue
        f_h = grid[mask] * h
        pos = np.interp(f_h, freqs, np.arange(len(freqs)))
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, len(freqs) - 1)
        frac = pos - lo
        amp = S[lo] * (1 - frac) + S[hi] * frac
        total[mask] += (1.0 / np.sqrt(h) if weight == "inv_sqrt"
                        else 1.0 / h) * amp
    return total / np.maximum(H.astype(np.float64) ** norm_pow, 1e-9)


def comb_wide(x, sr, lo=45.0, hi=800.0, df=0.5, **kw):
    grid = np.arange(lo, hi + df, df)
    S, fr = spec(x, sr)
    sc = comb_curve(S, fr, grid, **kw)
    i = int(np.argmax(sc))
    if 0 < i < len(grid) - 1:
        y0, y1, y2 = sc[i - 1], sc[i], sc[i + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
        if abs(d) > 1.0:
            d = 0.0
        return float(grid[i] + d * df)
    return float(grid[i])


def narrow_refine(S, freqs, center, lo_rel=0.85, hi_rel=1.20, df=0.2,
                  **kw):
    """候选附近的窄带精测，返回 (f0, 突出度)。"""
    g = np.arange(max(35.0, center * lo_rel),
                  min(1000.0, center * hi_rel) + df * 0.5, df)
    if len(g) < 5:
        return 0.0, 0.0
    sc = comb_curve(S, fr_res(freqs), g, **kw) if False else comb_curve(S, freqs, g, **kw)
    i = int(np.argmax(sc))
    peak = float(sc[i])
    med = float(np.median(sc)) + 1e-12
    prom = peak / med
    if 0 < i < len(g) - 1:
        y0, y1, y2 = sc[i - 1], sc[i], sc[i + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
        if abs(d) > 1.0:
            d = 0.0
        return float(g[i] + d * df), prom
    return float(g[i]), prom


def fr_res(freqs):
    return freqs


def measure_flow(x, sr, base=None, rel_lo=0.85, rel_hi=1.20,
                 divisors=(1, 2, 3, 4, 6, 8), base_tol=None):
    """完整测量流程。base 给定时额外做合理性约束。"""
    S, fr = spec(x, sr)
    f_raw = comb_wide(x, sr)
    if f_raw <= 0:
        return 0.0, {}
    cands = []
    for d in divisors:
        c = f_raw / d
        if 35.0 <= c <= 1000.0:
            cands.append(c)
    best = None
    sols = []
    for c in cands:
        f, prom = narrow_refine(S, fr, c, rel_lo, rel_hi)
        if f <= 0:
            continue
        ok = True
        if base and base > 0:
            r = f / base
            lo_r = base_tol[0] if base_tol else 0.45
            hi_r = base_tol[1] if base_tol else 2.40
            ok = (lo_r <= r <= hi_r)
        rec = {"cand": round(c, 1), "f": round(f, 1), "prom": round(prom, 2),
               "ok": ok}
        if ok:
            sols.append(rec)
    pool = sols if sols else []
    if not pool:
        return 0.0, {"f_raw": round(f_raw, 1), "sols": []}
    # 择一：突出度最高；若并列则选 f 最大的（真 f0 通常是最高的合法解）
    pool.sort(key=lambda r: (-r["prom"], -r["f"]))
    pick = pool[0]
    return pick["f"], {"f_raw": round(f_raw, 1), "sols": pool[:6],
                       "pick": pick}


# ================================================================== 评测
def eval_synth():
    print("=" * 92)
    print("A. 合成语音（真值精确）")
    print("=" * 92)
    x = synth_voice(120.0, seed=0)
    print("  [自检] f0=120: rms=%.4f  wide=%s"
          % (float(np.sqrt(np.mean(x.astype(np.float64) ** 2))),
             round(comb_wide(x, 24000), 1)))
    truths = [55, 65, 75, 85, 95, 110, 130, 160, 190, 230, 280, 340, 420]
    errs, oct_ok, fails = [], [], []
    for f0 in truths:
        for seed in (1, 2, 3):
            x = synth_voice(f0, seed=seed)
            f, det = measure_flow(x, 24000)
            if f <= 0:
                errs.append(200.0)
                fails.append((f0, seed, 0, det))
                continue
            e = abs(f / f0 - 1) * 100
            errs.append(e)
            oct_ok.append(e < 25)
            if e >= 25:
                fails.append((f0, seed, f, det))
    a = np.array(errs)
    print("  完整流程: 中位 %.2f%%  p75 %.2f%%  p90 %.2f%%  最大 %.2f%%  "
          "八度正确 %.1f%%"
          % (np.median(a), np.percentile(a, 75), np.percentile(a, 90),
             np.max(a), np.mean(oct_ok) * 100))
    # 只测宽测梳状谱作为对比
    a2 = []
    for f0 in truths:
        for seed in (1, 2, 3):
            x = synth_voice(f0, seed=seed)
            f = comb_wide(x, 24000)
            a2.append(abs(f / f0 - 1) * 100 if f > 0 else 200.0)
    a2 = np.array(a2)
    print("  仅宽测:   中位 %.2f%%  p90 %.2f%%  最大 %.2f%%  八度正确 %.1f%%"
          % (np.median(a2), np.percentile(a2, 90), np.max(a2),
             np.mean(a2 < 25) * 100))
    if fails:
        print("  --- 失败样本 ---")
        for f0, seed, f, det in fails[:10]:
            print("    f0=%5.1f seed=%d -> %6.1f  raw=%s sols=%s"
                  % (f0, seed, f, det.get("f_raw"),
                     json.dumps(det.get("sols", [])[:4])))
    return a


def eval_real():
    print()
    print("=" * 92)
    print("B. 真实人声：自一致性（对同一音频用不同处理后的相对关系）")
    print("=" * 92)
    files = sorted(glob.glob(f"{VOICES}/samples/*_voice.wav")) + \
            sorted(glob.glob(f"{VOICES}/samples_ext/*_voice.wav"))
    step = max(1, len(files) // 40)
    files = files[::step][:40]
    # 测原音频 f0（流程法），再降八度后重测，检验是否保持 x0.5 关系
    errs = []
    rows = []
    for p in files:
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        f0, det = measure_flow(x, sr)
        if f0 <= 0:
            continue
        y = shift(x, sr, 0.5)
        f1, _ = measure_flow(y, sr)
        if f1 <= 0:
            continue
        e = abs(f1 / (f0 * 0.5) - 1) * 100
        errs.append(e)
        rows.append((os.path.basename(p)[:12], f0, f1, e))
    a = np.array(errs)
    print("  降八度后跟踪: 中位 %.2f%%  p75 %.2f%%  p90 %.2f%%  最大 %.2f%%  "
          "八度正确 %.1f%%  (n=%d)"
          % (np.median(a), np.percentile(a, 75), np.percentile(a, 90),
             np.max(a), np.mean(a < 25) * 100, len(a)))
    rows.sort(key=lambda r: -r[3])
    for r in rows[:10]:
        print("    %-14s f0=%7.1f -> 降八度 %7.1f  误差 %6.2f%%" % r)


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


if __name__ == "__main__":
    eval_synth()
    eval_real()
