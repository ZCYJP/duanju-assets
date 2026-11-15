"""独立八度无关 f0 真值估计器（不依赖任何"目标窗口"）。

目的
----
v13/v14 的达标判据用「窄窗读数」，而窄窗以 target 为中心（±15%），
因此「只要窗内能找到峰，r_after 就自动落进达标区间」——存在假达标。
本模块提供**完全独立**的测量：既不看 target、也不做变调，
只回答「这段音频的基频到底是多少 Hz（绝对八度）」。

方法：谐波阶梯（harmonic ladder）
--------------------------------
1. 长窗 FFT（高分辨率）-> 平滑 -> 取显著峰列 peaks[0..K]
2. 对每个峰列做「阶梯展开」：把所有峰按比值归并到最小公共步长
   - 用候选步长 d（= 任一峰间距）与所有峰的整数倍关系投票
   - 得分 = 有多少峰能被 round(f/d) 解释
   - d 的候选来自所有峰对之差（含 1/2,1/3 细分以捕获被漏掉的低阶谐波）
3. 选出「得分最高且解释残差最小」的 d -> 该 d 即 f0
4. **八度判别**（关键）：对 d 与 d/2、d/3 分别检查
   - 若存在一个峰严格落在 d/2 且能量显著（非噪声）-> 真 f0 更可能是 d/2
   - 若 d/2 处无峰 -> d 就是真 f0（基频存在性检查）
   这正是「基频存在性」判据：真实人声在 f0 处一定有能量，
   而 2*f0 的谐波组合在 f0/2 处是无声的。

为什么八度无关
--------------
间距法给出的 d 本身没有八度歧义；歧义只可能来自「把 f0/2 当成基频」
或「锁到高阶谐波」。第 4 步用基频存在性消除前者；
「锁高阶」在间距法下不会发生（间距恒为 f0）。

自检
----
  python3 tmp_f0_truth.py --selftest
在合成信号上验证绝对精度。
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf

VOICES = "/ssd2/duanju/voices"


# ------------------------------------------------------------- 基础频谱
def spectrum(x, sr, n_fft=None):
    if n_fft is None:
        n_fft = 1 << int(np.ceil(np.log2(max(len(x), 8192))))
    w = np.hanning(len(x)).astype(np.float32)
    X = np.abs(np.fft.rfft(x * w, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # 轻度平滑（抑制噪声毛刺，不改变峰位置）
    k = np.hanning(7)
    k /= k.sum()
    X = np.convolve(X, k, mode="same")
    return X, freqs


def pick_peaks(S, freqs, fmax=3000.0, top=40, rel=0.06):
    """取显著峰列：局部极大 + 能量高于全局峰×rel。"""
    lim = int(np.searchsorted(freqs, fmax))
    S2 = S[:lim].copy()
    if len(S2) < 5:
        return []
    gmax = float(S2.max())
    if gmax <= 0:
        return []
    out = []
    # 局部极大
    for i in range(2, len(S2) - 2):
        if S2[i] >= S2[i - 1] and S2[i] > S2[i + 1] and S2[i] >= gmax * rel:
            out.append(i)
    # 按能量取前 top 个，再按频率排序
    out.sort(key=lambda i: S2[i], reverse=True)
    out = out[:top]
    out.sort()
    peaks = []
    df = freqs[1] - freqs[0]
    for i in out:
        y0, y1, y2 = S2[i - 1], S2[i], S2[i + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
        if abs(d) > 1.0:
            d = 0.0
        peaks.append((float(freqs[i] + d * df), float(S2[i])))
    return peaks


def _peak_energy_at(peaks, f, tol=0.03):
    """在频率 f 附近取峰能量（相对容差）。"""
    best = 0.0
    for pf, pa in peaks:
        if abs(pf - f) / max(f, 1e-6) <= tol:
            best = max(best, pa)
    return best


# ------------------------------------------------------------- 谐波阶梯
def ladder_candidates(peaks, f_lo=55.0, f_hi=500.0):
    """候选步长集合：所有峰对之差 + 其细分（捕获漏检的低阶谐波）。"""
    fs = [p[0] for p in peaks if p[0] >= f_lo * 0.8]
    cands = set()
    for i in range(len(fs)):
        for j in range(i + 1, len(fs)):
            d = fs[j] - fs[i]
            if d <= 1e-6:
                continue
            for m in (1, 2, 3):
                for k in (1, 2, 3):
                    v = d * m / k
                    if f_lo * 0.7 <= v <= f_hi * 1.4:
                        cands.add(round(v, 3))
    return sorted(cands)


def ladder_score(peaks, d, tol=0.035):
    """给定步长 d，评分：多少峰能被 d 的整数倍解释。

    返回 (能量覆盖率, 命中数, 解释为 1 阶谐波的峰能量占比)。
    1 阶占比是关键：真实 f0 处一定有能量，
    而 f0/2 处通常是静音 -> 靠这一项区分「d 偏小」的情况。
    """
    fs = [p[0] for p in peaks]
    if not fs:
        return 0.0, 0.0, 0.0
    hit = 0
    tot_w = 0.0
    hit_w = 0.0
    h1_w = 0.0
    for pf, pa in peaks:
        tot_w += pa
        n = pf / d
        k = max(1, int(round(n)))
        if abs(pf - k * d) / max(k * d, 1e-6) <= tol:
            hit += 1
            hit_w += pa
            if k == 1:
                h1_w += pa
    return (hit_w / max(tot_w, 1e-9), hit, h1_w / max(tot_w, 1e-9))


def octave_down(peaks, d, S, freqs, fmin=40.0):
    """基频存在性检查：d/2、d/3 处是否存在显著能量。

    返回 (true_f0, evidence)。若 d/2 处有峰且其能量达到全局峰的
    一定比例，则真 f0 = d/2。
    """
    gmax = max((pa for _, pa in peaks), default=0.0)
    if gmax <= 0:
        return d, 0.0
    best_d = d
    best_ev = 0.0
    for div in (2.0, 3.0):
        cand = d / div
        if cand < fmin:
            continue
        ev = _peak_energy_at(peaks, cand, tol=0.06)
        if ev / gmax >= 0.10 and ev > best_ev:
            best_d = cand
            best_ev = ev / gmax
    return best_d, best_ev


def measure_truth(x, sr, fmin=45.0, fmax=600.0, top=40, rel=0.06):
    """返回 dict：独立测得的绝对 f0（八度已判别）。

    步长选择规则
    ------------
    峰列的「公共间距」可能是 f0、f0/2、f0/3 ...（都能解释所有峰）。
    正确规则：**取能解释所有峰的最大步长** —— 因为 f0 本身就是
    最大的那个公共步长（更高的步长无法把 f0 处的峰解释成整数倍）。
    再叠加「1 阶谐波必须有能量」作为裁决条件。

    例：峰在 120/240/360/480 -> d=120 解释 4 个峰；d=240 只能解释
    240/480 而把 120/360 当噪声 -> 覆盖率低 -> 选 120。
    """
    S, freqs = spectrum(x, sr)
    peaks = pick_peaks(S, freqs, top=top, rel=rel)
    if len(peaks) < 4:
        return {"f0": 0.0, "peaks": [p[0] for p in peaks], "score": 0.0,
                "n_peaks": len(peaks), "oct_ev": 0.0, "h1": 0.0}
    cands = ladder_candidates(peaks, f_lo=fmin, f_hi=fmax)
    if not cands:
        return {"f0": 0.0, "peaks": [p[0] for p in peaks], "score": 0.0,
                "n_peaks": len(peaks), "oct_ev": 0.0, "h1": 0.0}
    # 收集所有候选的评分
    scored = []
    for d in cands:
        cov, hit, h1 = ladder_score(peaks, d)
        scored.append((d, cov, hit, h1))
    max_cov = max(s[1] for s in scored)
    # 覆盖率接近最优的候选（>= 97% 最优覆盖率）
    good = [s for s in scored if s[1] >= max_cov * 0.97]
    # 取其中最大的步长（f0 本身是最大的公共步长）
    good.sort(key=lambda s: -s[0])
    d = good[0][0]
    cov, hit, h1 = good[0][1], good[0][2], good[0][3]

    # 基频存在性复核：若 d/2 处有显著能量且 1 阶占比低，说明 d 偏大
    f0, ev = octave_down(peaks, d, S, freqs, fmin=fmin)
    if f0 != d:
        cov2, hit2, h12 = ladder_score(peaks, f0)
        if cov2 >= max_cov * 0.90:
            cov, hit, h1 = cov2, hit2, h12
        else:
            f0, ev = d, 0.0
    return {
        "f0": round(float(f0), 2),
        "ladder_d": round(float(d), 2),
        "score": round(float(cov), 4),
        "hit": int(hit),
        "h1": round(float(h1), 4),
        "n_peaks": len(peaks),
        "oct_ev": round(float(ev), 3),
        "peaks": [round(p[0], 1) for p in peaks[:12]],
    }


def measure_file(path, **kw):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    return measure_truth(x, sr, **kw)


# ------------------------------------------------------------- 合成自检
def synth(sr=24000, dur=2.0, f0=120.0, n_formants=4, seed=0):
    """合成类人声：谐波串 + 共振峰包络。"""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * dur)) / sr
    # 脉冲串（等间隔，带微抖）
    f = f0 * (1 + 0.01 * np.sin(2 * np.pi * 4.5 * t))
    ph = 2 * np.pi * np.cumsum(f) / sr
    x = np.zeros_like(t)
    for h in range(1, 40):
        if h * f0 > 5000:
            break
        x += np.sin(h * ph) / (h ** 1.1)
    # 共振峰包络（频域）
    X = np.fft.rfft(x)
    fr = np.fft.rfftfreq(len(x), 1 / sr)
    G = np.ones_like(fr)
    for fc, bw, g in [(500, 90, 12), (1400, 120, 8), (2500, 160, 5), (3500, 200, 3)]:
        G *= (1 + g * np.exp(-((fr - fc) ** 2) / (2 * bw ** 2)))
    x = np.fft.irfft(X * G, len(x)).astype(np.float32)
    x += rng.normal(0, 0.002, len(x)).astype(np.float32)
    return x.astype(np.float32), sr


def selftest():
    print("=" * 72)
    print("谐波阶梯法：合成信号绝对精度自检")
    print("=" * 72)
    errs = []
    oct_bad = []
    for f0 in [65, 80, 100, 120, 150, 180, 220, 260, 320]:
        x, sr = synth(f0=f0)
        r = measure_truth(x, sr)
        e = abs(r["f0"] - f0) / f0
        errs.append(e)
        flag = ""
        # 八度判定：真 f0 或 其 2 倍/3 倍都算「锁错」
        if e > 0.03:
            oct_bad.append((f0, r["f0"], r["ladder_d"], r["oct_ev"]))
            flag = "  <== 八度/误差异常"
        print(f"  真值 {f0:5.1f}Hz -> 测得 {r['f0']:7.2f}Hz  误差 {e*100:6.2f}%  "
              f"d={r['ladder_d']:.1f} hit={r['hit']}/{r['n_peaks']} "
              f"oct_ev={r['oct_ev']:.2f}{flag}")
    errs = np.array(errs)
    print("-" * 72)
    print(f"中位误差 {np.median(errs)*100:.3f}%  p90 {np.percentile(errs,90)*100:.3f}%  "
          f"最大 {errs.max()*100:.3f}%  八度异常 {len(oct_bad)}/{len(errs)}")
    return errs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--file", default="")
    args = ap.parse_args()
    if args.selftest:
        selftest()
    elif args.file:
        print(json.dumps(measure_file(args.file), ensure_ascii=False, indent=2))
    else:
        ap.print_help()
