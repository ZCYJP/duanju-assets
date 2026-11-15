"""尺度锚定 + 梳谱候选打分（v12 核心方法）。

修正 v11/GON 的两个缺陷
-----------------------
1. GON 用 max_ratio=1.85 约束候选 -> 把「需要修正 3 倍」的正确候选挡掉，
   逻辑自相矛盾（修正八度必须允许候选偏离）。
2. 绝对 f0 的八度本身无法确定（整体锁 2 倍 vs 1 倍，相对关系不变）。
   => 绝对值不重要，**相对关系才重要**。

尺度锚定（scale anchoring）
---------------------------
以基线的宽测值 base_raw 作为「任意尺度」（它本身可能锁错八度，无所谓）。
情绪音频的候选集合：
    C = { f_raw / k : k = 1..12 } ∩ [0.5*base_raw, 2.0*base_raw]
在 C 中，用**全频段梳状谱得分**（comb score）择一：
    真 f0 处所有谐波对齐，得分最高；
    2f0/3f0 处只有部分谐波对齐，且被 H^0.5 归一化惩罚。
再叠加 pyin 折八度候选作为补充票。

最终 r_i = c_i / base_raw，base_raw 的绝对八度无关紧要（分子分母同尺度）。
"""
import glob
import json
import os

import librosa
import numpy as np
import soundfile as sf

VOICES = "/ssd2/duanju/voices"
EMOTIONS = f"{VOICES}/emotions"
EMO_KEYS = ["xi", "nu", "ai", "le", "ju", "jing", "yan", "you", "xiu"]


# ------------------------------------------------------------- 梳谱
def comb_curve(S, freqs, grid, comb_max=2400.0, norm_pow=0.5):
    H = np.maximum((comb_max / grid).astype(int), 1)
    total = np.zeros(len(grid))
    for h in range(1, int(H.max()) + 1):
        mask = H >= h
        if not np.any(mask):
            continue
        f_h = grid[mask] * h
        pos = np.interp(f_h, freqs, np.arange(len(freqs)))
        lo_i = np.floor(pos).astype(int)
        hi_i = np.minimum(lo_i + 1, len(freqs) - 1)
        frac = pos - lo_i
        amp = S[lo_i] * (1 - frac) + S[hi_i] * frac
        total[mask] += amp / np.sqrt(h)
    return total / np.maximum(H.astype(np.float64) ** norm_pow, 1e-9)


def comb_score_at(S, freqs, f, comb_max=2400.0, norm_pow=0.5):
    """在任意 f 处计算梳谱得分（不限于网格）。"""
    if f <= 0:
        return 0.0
    H = max(int(comb_max / f), 1)
    total = 0.0
    for h in range(1, H + 1):
        fh = f * h
        if fh > freqs[-1]:
            break
        pos = fh / (freqs[1] - freqs[0])
        lo_i = int(pos)
        if lo_i >= len(freqs) - 1:
            break
        frac = pos - lo_i
        total += (S[lo_i] * (1 - frac) + S[lo_i + 1] * frac) / np.sqrt(h)
    return total / (H ** norm_pow)


def spec(x, sr):
    n_fft = 1 << int(np.ceil(np.log2(max(len(x), 4096))))
    w = np.hanning(len(x)).astype(np.float32)
    S = np.abs(np.fft.rfft(x * w, n=n_fft))
    return S, np.fft.rfftfreq(n_fft, 1.0 / sr)


def pyin_raw(x, sr):
    try:
        f, _, _ = librosa.pyin(x.astype(np.float32), fmin=45, fmax=800, sr=sr,
                               frame_length=2048, hop_length=512)
        v = f[(f > 45) & (f < 800)]
        return float(np.median(v)) if len(v) else 0.0
    except Exception:
        return 0.0


# ------------------------------------------------------------- 尺度锚定测量
def measure_scaled(x, sr, scale, win=(0.50, 2.00), ks=tuple(range(1, 13)),
                   df=0.5):
    """在尺度锚定下测 f0。scale 为任意尺度（基线宽测值）。"""
    S, fr = spec(x, sr)
    if S.max() <= 0:
        return 0.0, {}
    lo = max(35.0, scale * win[0])
    hi = min(1200.0, scale * win[1])

    # 宽测（梳谱全局峰）
    grid = np.arange(lo, hi + df, df)
    sc = comb_curve(S, fr, grid)
    i = int(np.argmax(sc))
    f_raw = float(grid[i])
    if 0 < i < len(grid) - 1:
        y0, y1, y2 = sc[i - 1], sc[i], sc[i + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
        if abs(d) > 1.0:
            d = 0.0
        f_raw = float(grid[i] + d * df)

    # 候选集
    cands = set()
    for k in ks:
        c = f_raw / k
        if lo <= c <= hi:
            cands.add(round(c, 3))
    p = pyin_raw(x, sr)
    if p > 0:
        for k in range(0, 6):
            c = p / (2 ** k)
            if lo <= c <= hi:
                cands.add(round(c, 3))
    if not cands:
        cands = {round(f_raw, 3)}

    # 梳谱得分择一
    scored = []
    for c in sorted(cands):
        s = comb_score_at(S, fr, c)
        scored.append((c, s))
    best = max(scored, key=lambda t: t[1])
    # 细化：最佳候选附近 ±4% 细扫
    g2 = np.arange(max(lo, best[0] * 0.96), min(hi, best[0] * 1.04) + df * 0.5,
                   df * 0.25)
    if len(g2) >= 5:
        s2 = comb_curve(S, fr, g2)
        j = int(np.argmax(s2))
        if 0 < j < len(g2) - 1:
            y0, y1, y2 = s2[j - 1], s2[j], s2[j + 1]
            den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
            if abs(d) > 1.0:
                d = 0.0
            f = float(g2[j] + d * (g2[1] - g2[0]))
        else:
            f = float(g2[j])
        return f, {"scale": round(scale, 1), "f_raw": round(f_raw, 1),
                   "best_cand": round(best[0], 1), "n_cand": len(cands),
                   "top": [(round(c, 1), round(s, 1))
                           for c, s in sorted(scored, key=lambda t: -t[1])[:4]]}
    return best[0], {"scale": round(scale, 1), "f_raw": round(f_raw, 1),
                     "best_cand": round(best[0], 1), "n_cand": len(cands)}


# ------------------------------------------------------------- 合成器
def synth_voice(f0, sr=24000, dur=2.0, seed=0, n_form=4):
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    t = np.arange(n) / sr
    src = np.zeros(n)
    period = sr / f0
    p = 0.0
    while p < n:
        i = int(p)
        L = max(4, int(period * 0.4))
        env = np.exp(-np.arange(L) / (L * 0.35))
        end = min(n, i + L)
        src[i:end] += env[:end - i]
        p += period
    src += 0.02 * rng.standard_normal(n)
    N = 1 << int(np.ceil(np.log2(n * 2)))
    X = np.fft.rfft(src, n=N)
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    fcs = [250, 700, 1220, 2600, 3400][:n_form + 1]
    bws = [80, 90, 110, 160, 200][:n_form + 1]
    G = np.zeros_like(freqs)
    for fc, bw in zip(fcs, bws):
        G += 10.0 / (1.0 + ((freqs - fc) / (bw * 1.5)) ** 2)
    G = np.maximum(G, 0.05)
    X = X * G
    x = np.fft.irfft(X, n=N)[:n].real
    env = np.clip(0.4 + 0.6 * np.sin(2 * np.pi * 2.1 * t + 0.9), 0.05, 1.0)
    x *= env
    pk = np.max(np.abs(x))
    if pk > 0 and np.isfinite(pk):
        x = x / pk * 0.5
    return np.nan_to_num(x).astype(np.float32)


# ------------------------------------------------------------- 评测
def eval_synth():
    print("=" * 92)
    print("A. 合成语音：尺度锚定能否恢复真 f0")
    print("=" * 92)
    x = synth_voice(120.0, seed=0)
    S, fr = spec(x, 24000)
    g = np.arange(45, 800.5, 0.5)
    print("  [自检] f0=120: comb_wide=%.1f  pyin=%.1f"
          % (g[int(np.argmax(comb_curve(S, fr, g)))], pyin_raw(x, 24000)))

    truths = [65, 85, 110, 160, 230, 340]
    rng = np.random.default_rng(11)
    errs, oct_ok = [], []
    for base_f0 in truths:
        for trial in range(2):
            base_true = base_f0
            mults = [1.0] + list(rng.uniform(0.82, 1.40, 4))
            trues = [base_true * m for m in mults]
            wavs, raws = [], []
            for t in trues:
                xs = synth_voice(t, seed=int(rng.integers(1, 9999)))
                wavs.append(xs)
                Sx, fx = spec(xs, 24000)
                gx = np.arange(45, 800.5, 0.5)
                raws.append(float(gx[int(np.argmax(comb_curve(Sx, fx, gx)))]))
            # 尺度 = 基线宽测值
            scale = raws[0]
            for idx in range(1, len(trues)):
                f, det = measure_scaled(wavs[idx], 24000, scale)
                if f <= 0:
                    errs.append(200.0)
                    oct_ok.append(False)
                    continue
                e = abs(f / trues[idx] - 1) * 100
                errs.append(e)
                oct_ok.append(e < 15)
    a = np.array(errs)
    print("  尺度锚定: 中位 %.2f%%  p75 %.2f%%  p90 %.2f%%  最大 %.2f%%  "
          "八度正确 %.1f%%  (n=%d)"
          % (np.median(a), np.percentile(a, 75), np.percentile(a, 90),
             np.max(a), np.mean(oct_ok) * 100, len(a)))
    return a


def _one_role(cid):
    d = os.path.join(EMOTIONS, cid)
    bp = glob.glob(f"{VOICES}/samples/{cid}_*_voice.wav") + \
         glob.glob(f"{VOICES}/samples_ext/{cid}_*_voice.wav")
    if not bp:
        return None
    bx, sr = sf.read(bp[0], dtype="float32")
    if bx.ndim > 1:
        bx = bx[:, 0]
    Sb, fb = spec(bx, sr)
    gb = np.arange(45, 800.5, 0.5)
    scale = float(gb[int(np.argmax(comb_curve(Sb, fb, gb)))])
    if scale <= 0:
        return None
    rr = {}
    for ek in EMO_KEYS:
        ep = glob.glob(f"{d}/*_{ek}.wav")
        if not ep:
            continue
        ex, _ = sf.read(ep[0], dtype="float32")
        if ex.ndim > 1:
            ex = ex[:, 0]
        f, det = measure_scaled(ex, sr, scale)
        rr[ek] = round(f / scale, 3) if f > 0 else 0.0
    if len(rr) != 9:
        return None
    return cid, round(scale, 1), rr


def eval_real():
    print()
    print("=" * 92)
    print("B. 真实人声：150 角色，尺度锚定后的 9 情绪倍率")
    print("=" * 92)
    from concurrent.futures import ProcessPoolExecutor
    cids = [c for c in sorted(os.listdir(EMOTIONS))
            if os.path.isdir(os.path.join(EMOTIONS, c))]
    rows = []
    with ProcessPoolExecutor(max_workers=48) as ex:
        for r in ex.map(_one_role, cids):
            if r:
                rows.append(r)
    all_r = [v for _, _, rr in rows for v in rr.values()]
    a = np.array(all_r)
    inrange = np.mean((a >= 0.80) & (a <= 1.45)) * 100
    print("  倍率分布: 中位 %.3f  p5 %.3f  p95 %.3f" %
          (np.median(a), np.percentile(a, 5), np.percentile(a, 95)))
    print("  落在 0.80~1.45 的比例: %.1f%%   (n=%d)" % (inrange, len(a)))
    print("  >1.45 的: %.1f%%   <0.80 的: %.1f%%"
          % (np.mean(a > 1.45) * 100, np.mean(a < 0.80) * 100))
    print("\n  --- 前 12 个角色 ---")
    for cid, scale, rr in rows[:12]:
        print("    %-6s scale=%6.1f | %s"
              % (cid, scale, " ".join("%s=%.2f" % (k, rr[k]) for k in EMO_KEYS)))
    return rows


if __name__ == "__main__":
    eval_synth()
    eval_real()
