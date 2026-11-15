"""组内联合八度归一化（Group Octave Normalization, GON）。

思路转变
--------
此前 8 代（v3..v10）都在追求「单条音频的通用精确 f0 估计」，实测证明这在低音区
物理上做不到：
  - 真实人声 pyin 会锁到 3~12 倍高次谐波（M02_yan 真 ~90Hz 读 920Hz）
  - 合成语音上梳状谱也会偏（强共振峰抬高高次谐波）
  - 变调算子本身却是精确的（降调误差 0.1~1.1%）
=> 结论：唯一可靠的「真值锚」不是频谱，而是**组内结构一致性**。

结构事实（强约束）
------------------
同一角色的 1 条基线 + 9 条情绪音频出自同一 TTS 音色设计，
真实 f0 必然同量级；用户给的 9 情绪音高倍率表也限定在基线的 0.82~1.40 倍。
=> 全组 10 条的 f0 相对中位数的比值必落在约 [0.7, 1.7]。

算法
----
1. 每条音频宽测得 f_raw（可能差整数倍八度）
2. 候选集 C_i = { f_raw_i / k : k=1..10 } ∪ { f_raw_i * 2 }
3. 迭代（初值取各条 f_raw 的对数中位数 μ）：
     对每条 i，在 C_i 中选使 |log(c) - μ| 最小的候选
     再更新 μ = median(log(c_i))
   收敛后即得全组一致的 f0 集合
4. base = 基线那条的 f0；r_i = f0_i / base

评估
----
A. 合成语音（真值精确）：组内八度归一后能否恢复真 f0
B. 真实人声：联合估计与「单条宽测」的对比 + 组内离散度
"""
import glob
import itertools
import json
import os

import librosa
import numpy as np
import soundfile as sf

VOICES = "/ssd2/duanju/voices"
EMOTIONS = f"{VOICES}/emotions"
EMO_KEYS = ["xi", "nu", "ai", "le", "ju", "jing", "yan", "you", "xiu"]


# ------------------------------------------------------------- 合成器
def synth_voice(f0, sr=24000, dur=2.0, seed=0, n_form=4):
    """源-滤波器合成语音（时域脉冲串 + 频域共振峰包络，稳定）。"""
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    t = np.arange(n) / sr
    # 脉冲串（每个周期一个衰减脉冲），严格按周期推进
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
    # 频域共振峰包络
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
    # 音节包络
    env = np.clip(0.4 + 0.6 * np.sin(2 * np.pi * 2.1 * t + 0.9), 0.05, 1.0)
    x *= env
    pk = np.max(np.abs(x))
    if pk > 0 and np.isfinite(pk):
        x = x / pk * 0.5
    return np.nan_to_num(x).astype(np.float32)


# ------------------------------------------------------------- 宽测
def comb_wide(x, sr, lo=45.0, hi=800.0, df=0.5):
    n_fft = 1 << int(np.ceil(np.log2(max(len(x), 4096))))
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
        pos = np.interp(f_h, freqs, np.arange(len(freqs)))
        lo_i = np.floor(pos).astype(int)
        hi_i = np.minimum(lo_i + 1, len(freqs) - 1)
        frac = pos - lo_i
        amp = S[lo_i] * (1 - frac) + S[hi_i] * frac
        total[mask] += amp / np.sqrt(h)
    total /= np.maximum(H.astype(np.float64) ** 0.5, 1e-9)
    i = int(np.argmax(total))
    if 0 < i < len(grid) - 1:
        y0, y1, y2 = total[i - 1], total[i], total[i + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
        if abs(d) > 1.0:
            d = 0.0
        return float(grid[i] + d * df)
    return float(grid[i])


def pyin_f0(x, sr):
    try:
        f, _, _ = librosa.pyin(x.astype(np.float32), fmin=45, fmax=800, sr=sr,
                               frame_length=2048, hop_length=512)
        v = f[(f > 45) & (f < 800)]
        return float(np.median(v)) if len(v) else 0.0
    except Exception:
        return 0.0


# ------------------------------------------------------------- 组内联合归一
def gon(f_raws, max_ratio=1.85, iters=12, ks=tuple(range(1, 11))):
    """组内八度归一化。

    f_raws: 各条音频的宽测值
    返回: (归一后的 f0 列表, 诊断)
    """
    f_raws = [f for f in f_raws if f and f > 0]
    if not f_raws:
        return [], {}
    cands = []
    for f in f_raws:
        cs = sorted({round(f / k, 4) for k in ks if 30.0 <= f / k <= 1200.0}
                    | {round(f * 2, 4)} if 30.0 <= f * 2 <= 1200.0 else set())
        if not cs:
            cs = [f]
        cands.append(cs)

    # 初始化：取各条 f_raw 的对数中位数
    mu = float(np.median(np.log(f_raws)))
    cur = list(f_raws)
    for _ in range(iters):
        new = []
        for cs in cands:
            # 选使 |log(c) - mu| 最小且在 [mu/max_ratio, mu*max_ratio] 内的
            best, bd = None, 1e18
            for c in cs:
                lc = np.log(c)
                if lc < mu - np.log(max_ratio) or lc > mu + np.log(max_ratio):
                    continue
                d = abs(lc - mu)
                if d < bd:
                    bd, best = d, c
            if best is None:
                best = min(cs, key=lambda c: abs(np.log(c) - mu))
            new.append(best)
        nmu = float(np.median(np.log(new)))
        if abs(nmu - mu) < 1e-6 and all(abs(a - b) < 1e-6
                                        for a, b in zip(new, cur)):
            mu = nmu
            cur = new
            break
        mu = nmu
        cur = new
    ratios = np.exp(np.log(cur) - mu)
    return cur, {"mu": float(np.exp(mu)),
                 "ratio_spread": float(ratios.max() - ratios.min()),
                 "ratio_max": float(ratios.max()),
                 "ratio_min": float(ratios.min())}


# ------------------------------------------------------------- 评测 A：合成
def eval_synth():
    print("=" * 92)
    print("A. 合成语音：组内归一能否恢复真 f0")
    print("=" * 92)
    # 自检：合成器是否可测
    x = synth_voice(120.0, seed=0)
    print("  [自检] f0=120 合成：wide=%.1f  pyin=%.1f"
          % (comb_wide(x, 24000), pyin_f0(x, 24000)))

    truths = [55, 65, 75, 85, 95, 110, 130, 160, 190, 230, 280, 340, 420]
    # 每组：1 条基线 + 9 条情绪，真实 f0 在基线的 0.82~1.40 倍
    rng = np.random.default_rng(3)
    all_single, all_group = [], []
    for base_f0 in truths:
        for trial in range(3):
            base_true = base_f0
            mults = [1.0] + list(rng.uniform(0.82, 1.40, 9))
            trues = [base_true * m for m in mults]
            raws = []
            for t in trues:
                xs = synth_voice(t, seed=int(rng.integers(1, 999)))
                raws.append(comb_wide(xs, 24000))
            # 单条估计（对照组）：直接用宽测值比基线宽测值
            single = abs(raws[1] / raws[0] - mults[1]) / mults[1] * 100
            all_single.append(single)
            # 组内归一
            ests, det = gon(raws)
            if len(ests) == len(trues):
                e = abs(ests[1] / ests[0] - mults[1]) / mults[1] * 100
                all_group.append(e)
    s = np.array(all_single)
    g = np.array(all_group)
    print("  单条宽测比值:  中位 %.2f%%  p90 %.2f%%  八度正确 %.1f%%"
          % (np.median(s), np.percentile(s, 90), np.mean(s < 15) * 100))
    print("  组内归一:      中位 %.2f%%  p90 %.2f%%  八度正确 %.1f%%"
          % (np.median(g), np.percentile(g, 90), np.mean(g < 15) * 100))
    return s, g


# ------------------------------------------------------------- 评测 B：真实
def eval_real():
    print()
    print("=" * 92)
    print("B. 真实人声：150 角色组内归一")
    print("=" * 92)
    base_map = json.load(open(f"{VOICES}/base_f0_v11.json"))
    roles = sorted(base_map.keys())
    singles, groups = [], []
    spreads = []
    rows = []
    for cid in roles:
        paths = []
        bp = glob.glob(f"{VOICES}/samples/{cid}_*_voice.wav") + \
             glob.glob(f"{VOICES}/samples_ext/{cid}_*_voice.wav")
        if not bp:
            continue
        paths.append(bp[0])
        for ek in EMO_KEYS:
            ep = glob.glob(f"{EMOTIONS}/{cid}/*_{ek}.wav")
            if ep:
                paths.append(ep[0])
        if len(paths) < 5:
            continue
        raws = []
        for p in paths:
            x, sr = sf.read(p, dtype="float32")
            if x.ndim > 1:
                x = x[:, 0]
            raws.append(comb_wide(x, sr))
        if not raws or not raws[0]:
            continue
        # 单条：宽测比值（与 v7 口径相同的量纲）
        singles.extend(abs(r / raws[0] - 1) for r in raws[1:])
        ests, det = gon(raws)
        spreads.append(det.get("ratio_spread", 0))
        if len(ests) == len(paths):
            groups.extend(abs(e / ests[0] - 1) for e in ests[1:])
            rows.append((cid, raws[0], ests[0], det.get("ratio_max"),
                         det.get("ratio_min")))
    s = np.array(singles)
    g = np.array(groups)
    print("  单条宽测比值偏离 1:  中位 %.1f%%  p90 %.1f%%  >25%%占比 %.1f%%"
          % (np.median(s), np.percentile(s, 90), np.mean(s > 25) * 100))
    print("  组内归一比值偏离 1:  中位 %.1f%%  p90 %.1f%%  >25%%占比 %.1f%%"
          % (np.median(g), np.percentile(g, 90), np.mean(g > 25) * 100))
    print("  组内倍率跨度: 中位 %.3f  p90 %.3f"
          % (np.median(spreads), np.percentile(spreads, 90)))
    # 看归一后每个角色的 9 情绪倍率
    print("\n  --- 归一后 9 情绪倍率（应落在 0.82~1.40）---")
    for cid in roles[:12]:
        paths = []
        bp = glob.glob(f"{VOICES}/samples/{cid}_*_voice.wav") + \
             glob.glob(f"{VOICES}/samples_ext/{cid}_*_voice.wav")
        if not bp:
            continue
        paths.append(bp[0])
        keys = ["base"]
        for ek in EMO_KEYS:
            ep = glob.glob(f"{EMOTIONS}/{cid}/*_{ek}.wav")
            if ep:
                paths.append(ep[0])
                keys.append(ek)
        raws = []
        for p in paths:
            x, sr = sf.read(p, dtype="float32")
            if x.ndim > 1:
                x = x[:, 0]
            raws.append(comb_wide(x, sr))
        ests, det = gon(raws)
        if len(ests) != len(paths):
            continue
        rr = [e / ests[0] for e in ests]
        print("    %-6s raw_base=%6.1f 归一base=%6.1f | %s"
              % (cid, raws[0], ests[0],
                 " ".join("%s=%.2f" % (k, v) for k, v in zip(keys[1:], rr[1:]))))


if __name__ == "__main__":
    eval_synth()
    eval_real()
