"""f0 八度消歧策略评测台（有精确真值）。

背景
----
低 f0 区的「通用精确 f0 估计」物理上做不到：
  - 80Hz 的语音，2f=160Hz 的谐波能量常压过基频
  - pyin 在真实人声上会读出 3~12 倍的荒谬值（M02_yan: 真 90Hz 读 920Hz）
=> 必须换思路：不是「更准地估计 f0」，而是「用结构约束消除歧义」。

策略空间
--------
S1 自相关长窗          : 8192 窗 ACF 峰值 + 抛物线细化
S2 YIN 绝对阈值        : 首个低于阈值的局部极小
S3 梳状谱连续峰        : 谐波梳在频域找峰
S4 梳状谱 + 窄带自洽   : 候选八度枚举 -> 每个候选处窄带梳峰锐度打分
S5 S4 + 时域周期自洽   : 候选处归一化自相关作为附加分
S6 多策略投票          : S1..S5 输出的八度投票

评测
----
1) 合成源-滤波器语音（真值精确）：覆盖 50~450Hz，含颤音/共振峰/噪声
2) 真实人声 round-trip（相对精度）
"""
import glob
import json
import os

import librosa
import numpy as np
import soundfile as sf

VOICES = "/ssd2/duanju/voices"


# ================================================================== 合成
def synth_voice(f0, sr=24000, dur=3.0, seed=0, jitter=0.02,
                formants=((700, 90), (1220, 110), (2600, 160), (3400, 200)),
                f0_drift=0.03):
    """源-滤波器合成语音：脉冲串激励 + 共振峰 + 颤音 + 噪声。"""
    rng = np.random.default_rng(seed)
    n = int(sr * dur)
    t = np.arange(n) / sr
    # 颤音 + 漂移
    vib = 1.0 + f0_drift * np.sin(2 * np.pi * 4.7 * t + rng.uniform(0, 6.28))
    phase = 2 * np.pi * np.cumsum(f0 * vib) / sr
    src = np.zeros(n)
    # 声门脉冲串（单调衰减脉冲；用累加器避免浮点漂移）
    period = sr / f0
    pos = 0.0
    while pos < n:
        p = int(pos)
        L = max(3, int(period * 0.35))
        env = (1.0 - np.arange(L) / L) ** 2
        end = min(n, p + L)
        src[p:end] += env[:end - p]
        pos += period * (1.0 + rng.uniform(-0.01, 0.01))
    # 噪声激励（送气感）
    src += 0.03 * rng.standard_normal(n)
    # 共振峰滤波（二阶带通；用 scipy.signal.lfilter 等价的稳定递推）
    x = src.astype(np.float64)
    for fc, bw in formants:
        r = np.exp(-np.pi * bw / sr)
        th = 2 * np.pi * fc / sr
        a1 = -2 * r * np.cos(th)
        a2 = r * r
        b0 = (1 - r) * np.sqrt(1 - 2 * r * np.cos(2 * th) + r * r)
        y = np.zeros_like(x)
        y1 = 0.0
        y2 = 0.0
        for i in range(n):
            yv = b0 * x[i] + a1 * y1 + a2 * y2
            y2 = y1
            y1 = yv
            y[i] = yv
        # 每级归一化，防止级联溢出
        mx = np.max(np.abs(y))
        if mx > 0 and np.isfinite(mx):
            y = y / mx
        x = y
    x += 0.004 * rng.standard_normal(n)
    pk = np.max(np.abs(x))
    if pk > 0 and np.isfinite(pk):
        x = x / pk * 0.55
    x = np.nan_to_num(x)
    return x.astype(np.float32)


# ================================================================== 估计器
def ac_long(x, sr, frame=8192):
    """长窗归一化自相关峰。"""
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    seg = x[:frame].astype(np.float64)
    seg = seg - seg.mean()
    n = len(seg)
    acf = np.correlate(seg, seg, "full")[n - 1:]
    lo = max(1, int(sr / 800))
    hi = min(n - 1, int(sr / 45))
    if hi <= lo + 2:
        return 0.0
    s = acf[lo:hi]
    e = acf[0]
    if e <= 0:
        return 0.0
    s = s / e
    i = int(np.argmax(s))
    # 抛物线细化
    if 0 < i < len(s) - 1:
        y0, y1, y2 = s[i - 1], s[i], s[i + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
        if abs(d) > 1:
            d = 0.0
        tau = lo + i + d
    else:
        tau = lo + i
    return float(sr / tau) if tau > 0 else 0.0


def yin_abs(x, sr, frame=8192, threshold=0.15):
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    seg = x[:frame].astype(np.float64)
    seg = seg - seg.mean()
    n = len(seg)
    tau_min = max(int(sr / 800), 2)
    tau_max = min(int(sr / 45), n - 1)
    if tau_max <= tau_min + 2:
        return 0.0
    acf = np.correlate(seg, seg, "full")[n - 1:]
    sq = np.concatenate(([0.0], np.cumsum(seg ** 2)))
    ptau = sq[n:] - sq[:n]
    d = np.maximum(float(np.dot(seg, seg)) + ptau[:tau_max + 1]
                   - 2.0 * acf[:tau_max + 1], 0.0)
    cum = np.cumsum(d[1:])
    taus = np.arange(1, len(d))
    dp = np.ones_like(d)
    dp[1:] = d[1:] * taus / np.maximum(cum, 1e-12)
    below = np.where(dp[tau_min:tau_max + 1] < threshold)[0]
    if len(below):
        i = int(below[0]) + tau_min
        while i + 1 <= tau_max and dp[i + 1] < dp[i]:
            i += 1
    else:
        i = int(np.argmin(dp[tau_min:tau_max + 1])) + tau_min
    if 0 < i < len(dp) - 1:
        y0, y1, y2 = dp[i - 1], dp[i], dp[i + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
        if abs(d) > 1:
            d = 0.0
        tau = i + d
    else:
        tau = i
    return float(sr / tau) if tau > 0 else 0.0


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


def comb_peak(x, sr, lo=45.0, hi=800.0, df=0.5, weight="inv_sqrt",
              norm_pow=0.5):
    grid = np.arange(lo, hi + df, df)
    S, fr = spec(x, sr)
    sc = comb_curve(S, fr, grid, weight=weight, norm_pow=norm_pow)
    i = int(np.argmax(sc))
    y0, y1, y2 = sc[i - 1], sc[i], sc[i + 1]
    den = y0 - 2 * y1 + y2
    d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
    if abs(d) > 1.0:
        d = 0.0
    return float(grid[i] + d * df), sc, grid


def narrow_sharpness(x, sr, f0, rel=0.12):
    """候选 f0 处的窄带梳峰锐度：真 f0 处梳峰应明显尖于其八度邻域。

    做法：在 f0 附近细扫，得局部峰；再在 f0*2、f0/2 附近同样细扫；
    真 f0 的峰高 / 邻域八度峰高 的比值应最大。
    """
    S, fr = spec(x, sr)
    out = {}
    for tag, c in (("f", f0), ("2f", f0 * 2), ("f/2", f0 / 2)):
        if c < 40 or c > 900:
            out[tag] = 0.0
            continue
        g = np.arange(max(40.0, c * 0.94), min(1000.0, c * 1.06) + 0.25, 0.25)
        if len(g) < 5:
            out[tag] = 0.0
            continue
        sc = comb_curve(S, fr, g)
        out[tag] = float(np.max(sc))
    return out


def ac_at(x, sr, f0, frame=8192):
    """候选 f0 处的归一化自相关（周期性强度）。"""
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
    e = acf[0]
    if e <= 0:
        return 0.0
    a = acf[t]
    b = 0.5 * (acf[t - 1] + acf[t + 1])
    return float((a - b) / e)


# ================================================================== 策略
def strat_s1(x, sr):
    return ac_long(x, sr)


def strat_s2(x, sr):
    return yin_abs(x, sr)


def strat_s3(x, sr):
    f, _, _ = comb_peak(x, sr)
    return f


def strat_s4(x, sr):
    """梳状谱峰 + 八度候选窄带锐度择一。"""
    f, _, _ = comb_peak(x, sr)
    if f <= 0:
        return 0.0
    cands = [f, f / 2, f / 4, f * 2]
    best, bs = f, -1.0
    for c in cands:
        if c < 45 or c > 800:
            continue
        sh = narrow_sharpness(x, sr, c)
        s = sh["f"] / (max(sh["2f"], 1e-9) + max(sh["f/2"], 1e-9))
        if s > bs:
            bs, best = s, c
    return best


def strat_s5(x, sr):
    """S4 + 时域周期性加权。"""
    f, _, _ = comb_peak(x, sr)
    if f <= 0:
        return 0.0
    cands = [f, f / 2, f / 4, f * 2]
    best, bs = f, -1e9
    for c in cands:
        if c < 45 or c > 800:
            continue
        sh = narrow_sharpness(x, sr, c)
        ratio = sh["f"] / (max(sh["2f"], 1e-9) + max(sh["f/2"], 1e-9))
        acf = ac_at(x, sr, c)
        s = ratio * (0.5 + acf)
        if s > bs:
            bs, best = s, c
    return best


STRATS = {
    "S1_ac_long": strat_s1,
    "S2_yin_abs": strat_s2,
    "S3_comb": strat_s3,
    "S4_comb_narrow": strat_s4,
    "S5_comb_narrow_ac": strat_s5,
}


# ================================================================== 评测
def eval_synth():
    print("=" * 92)
    print("A. 合成源-滤波器语音（真值精确）")
    print("=" * 92)
    # 自检：合成信号本身要被各估计器正确读出
    x = synth_voice(120.0, seed=0)
    print("  [自检] f0=120 合成信号： rms=%.4f  ac_long=%.1f  yin=%.1f  comb=%.1f"
          % (float(np.sqrt(np.mean(x ** 2))), ac_long(x, 24000),
             yin_abs(x, 24000), comb_peak(x, 24000)[0]))
    truths = [55, 65, 75, 85, 95, 110, 130, 160, 190, 230, 280, 340, 420]
    acc = {k: [] for k in STRATS}
    for f0 in truths:
        for seed in (1, 2):
            x = synth_voice(f0, seed=seed)
            for name, fn in STRATS.items():
                est = fn(x, 24000)
                if est > 0:
                    acc[name].append(abs(est / f0 - 1) * 100)
    print("%-20s %8s %8s %8s %8s  %s" %
          ("策略", "中位", "p75", "p90", "最大", "八度正确率(<25%)"))
    for name, arr in acc.items():
        a = np.array(arr)
        oct_ok = np.mean(a < 25) * 100
        print("%-20s %7.2f%% %7.2f%% %7.2f%% %7.2f%%  %5.1f%%"
              % (name, np.median(a), np.percentile(a, 75),
                 np.percentile(a, 90), np.max(a), oct_ok))
    return acc


def eval_real():
    print()
    print("=" * 92)
    print("B. 真实人声：降八度后能否正确跟踪（真值 = 原读数 x 0.5）")
    print("=" * 92)
    files = sorted(glob.glob(f"{VOICES}/samples/*_voice.wav")) + \
            sorted(glob.glob(f"{VOICES}/samples_ext/*_voice.wav"))
    step = max(1, len(files) // 30)
    files = files[::step][:30]
    acc = {k: [] for k in STRATS}
    for p in files:
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        # 用 S4 定原 f0（作为参考真值）
        f0 = strat_s4(x, sr)
        if f0 <= 0:
            continue
        # 降八度
        y = resample_shift(x, sr, 0.5)
        truth = f0 * 0.5
        for name, fn in STRATS.items():
            est = fn(y, sr)
            if est > 0:
                acc[name].append(abs(est / truth - 1) * 100)
    print("%-20s %8s %8s %8s %8s  %s" %
          ("策略", "中位", "p75", "p90", "最大", "八度正确率(<25%)"))
    for name, arr in acc.items():
        if not arr:
            continue
        a = np.array(arr)
        oct_ok = np.mean(a < 25) * 100
        print("%-20s %7.2f%% %7.2f%% %7.2f%% %7.2f%%  %5.1f%%"
              % (name, np.median(a), np.percentile(a, 75),
                 np.percentile(a, 90), np.max(a), oct_ok))


def resample_shift(x, sr, factor):
    """降八度变调（保持时长）：重采样 + 相位声码器。"""
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
    a = eval_synth()
    eval_real()
    json.dump({k: {"median": float(np.median(v)),
                   "p90": float(np.percentile(v, 90)),
                   "max": float(np.max(v))}
               for k, v in a.items() if v},
              open(f"{VOICES}/f0_strategy_lab.json", "w"), indent=2)
