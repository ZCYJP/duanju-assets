"""基线锚定的 f0 估计器（v11 方案的实验室）。

核心问题
--------
对超低音男声，所有估计器都会锁到 2f/3f/4f 高次谐波：
    M28_nu 真实谐波间距 ~130Hz，pyin 读 295.8Hz（2.27x）
结果：回正时把本已正常的音频降调 3~4 倍（音质灾难）。

关键结构事实
------------
情绪音频与角色基线是同一 TTS 音色设计的产物，音高必然同量级。
用户给的 9 情绪音高倍率表也明确限定在基线的 0.82~1.40 倍。
=> 把 f0 搜索限制在 [base*0.55, base*2.3] 的窄带内，
   倍频歧义被搜索带本身消除（带宽 < 2 个八度，且目标落在带中心）。

方法
----
1) 谱：整段 FFT + 逐帧 FFT，两级
2) 打分：谐波梳 score(f) = Σ_h w_h · S(h·f) / H^norm
   - w_h = 1/sqrt(h) 或 1/h
   - 归一化消除「低 f 收集更多谐波」的偏置
3) 逐帧打分 -> 取逐帧 argmax 的中位数（抗颤音）
4) 带内择优 + 八度候选投票

验证
----
A. 真实基线音频 × 已知变调倍率 的 round-trip（决定性）
B. 失败案例的锚定读数 vs pyin 读数
"""
import glob
import json
import os

import librosa
import numpy as np
import soundfile as sf

VOICES = "/ssd2/duanju/voices"
EMOTIONS = f"{VOICES}/emotions"

# 带内测量（默认）
BAND_LO, BAND_HI = 0.55, 2.30


# ------------------------------------------------------------------ 谱
def frame_spectrum(x, sr, n_fft=4096, hop=1024):
    """逐帧幅度谱 -> (frames, bins), freqs"""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    fr = librosa.util.frame(x, frame_length=n_fft, hop_length=hop)
    w = np.hanning(n_fft).astype(np.float32)
    S = np.abs(np.fft.rfft(fr * w[:, None], axis=0))  # (bins, frames)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    return S.T, freqs  # (frames, bins)


def whole_spectrum(x, sr):
    n_fft = 1 << int(np.ceil(np.log2(max(len(x), 4096))))
    w = np.hanning(len(x)).astype(np.float32)
    X = np.abs(np.fft.rfft(x * w, n=n_fft))
    return X, np.fft.rfftfreq(n_fft, 1.0 / sr)


# ------------------------------------------------------------------ 梳打分
def comb_score_grid(S, freqs, grid, comb_max, weight, norm_pow):
    """S: (frames, bins) 或 (bins,)。返回 (grid 上的得分, frames 数)。"""
    if S.ndim == 1:
        S = S[None, :]
        one = True
    else:
        one = False
    H = np.maximum((comb_max / grid).astype(int), 1)
    hmax = int(H.max())
    # 采样帧（最多 40 帧，避免过慢）
    if S.shape[0] > 40:
        idx = np.linspace(0, S.shape[0] - 1, 40).astype(int)
        S = S[idx]
    total = np.zeros((S.shape[0], len(grid)))
    for h in range(1, hmax + 1):
        mask = H >= h
        if not np.any(mask):
            continue
        f_h = grid[mask] * h
        # 逐帧插值
        pos = np.interp(f_h, freqs, np.arange(len(freqs)))
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, len(freqs) - 1)
        frac = pos - lo
        amp = S[:, lo] * (1 - frac) + S[:, hi] * frac   # (frames, nmask)
        total[:, mask] += (1.0 / np.sqrt(h) if weight == "inv_sqrt"
                           else 1.0 / h) * amp
    denom = np.maximum(H.astype(np.float64) ** norm_pow, 1e-9)
    score = total / denom[None, :]
    if one:
        return score[0]
    return score


def refine_peak(grid, score):
    i = int(np.argmax(score))
    if i <= 0 or i >= len(grid) - 1:
        return float(grid[i]), float(score[i]), i
    y0, y1, y2 = score[i - 1], score[i], score[i + 1]
    den = y0 - 2 * y1 + y2
    d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
    if abs(d) > 1.0:
        d = 0.0
    return float(grid[i] + d * (grid[1] - grid[0])), float(y1), i


def f0_anchored(x, sr, base, band=(BAND_LO, BAND_HI),
                weight="inv_sqrt", norm_pow=0.5, df=0.5,
                comb_max=2200.0, per_frame=True):
    """基线锚定的 f0 估计：只在 [base*band_lo, base*band_hi] 内搜。"""
    lo = max(40.0, base * band[0])
    hi = min(1200.0, base * band[1])
    if hi <= lo:
        return 0.0, {}
    grid = np.arange(lo, hi + df, df)

    if per_frame:
        S, freqs = frame_spectrum(x, sr)
        sc = comb_score_grid(S, freqs, grid, comb_max, weight, norm_pow)
        # 逐帧 argmax 的中位数（抗颤音/加噪）
        idx = np.argmax(sc, axis=1)
        vals = grid[idx]
        f_med = float(np.median(vals))
        # 在中位数附近 ±12% 再做一次整段梳打分细化
        g2 = np.arange(max(lo, f_med * 0.88), min(hi, f_med * 1.12) + df * 0.5, df * 0.25)
        if len(g2) >= 5:
            Sw, fw = whole_spectrum(x, sr)
            sc2 = comb_score_grid(Sw, fw, g2, comb_max, weight, norm_pow)
            f, _, _ = refine_peak(g2, sc2)
            return f, {"f_frame_med": f_med, "mode_frames": int(len(vals))}
        return f_med, {"f_frame_med": f_med, "mode_frames": int(len(vals))}

    Sw, fw = whole_spectrum(x, sr)
    sc = comb_score_grid(Sw, fw, grid, comb_max, weight, norm_pow)
    f, s, _ = refine_peak(grid, sc)
    return f, {"score": round(float(s), 3)}


def f0_anchored_multi(x, sr, base, band=(BAND_LO, BAND_HI)):
    """多配置组合（梳谱 3 组 + pyin 带内），取中位数。"""
    outs = []
    for kw in (dict(weight="inv_sqrt", norm_pow=0.5),
               dict(weight="inv", norm_pow=0.0),
               dict(weight="inv_sqrt", norm_pow=0.3)):
        f, _ = f0_anchored(x, sr, base, band=band, **kw)
        if f > 0:
            outs.append(f)
    # pyin 带内（把 pyin 原始读数按八度折回带内）
    try:
        fp, _, _ = librosa.pyin(x.astype(np.float32), fmin=40, fmax=1200, sr=sr,
                                frame_length=2048, hop_length=512)
        v = fp[(fp > 40) & (fp < 1200)]
        if len(v):
            p = float(np.median(v))
            lo = base * band[0]
            hi = base * band[1]
            k = 0
            while p / (2 ** (k + 1)) >= lo:
                k += 1
            while p / (2 ** k) > hi and k < 6:
                k += 1
            outs.append(p / (2 ** k))
    except Exception:
        pass
    if not outs:
        return 0.0, {}
    return float(np.median(outs)), {"n_cfg": len(outs),
                                    "vals": [round(o, 1) for o in outs]}


# ------------------------------------------------------------------ 变调
def pitch_shift_once(x, sr, factor):
    if abs(factor - 1.0) < 0.005:
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
    elif len(out) > n_in:
        out = out[:n_in]
    rms_out = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    if rms_out > 1e-8 and rms_in > 1e-8:
        out = out * (rms_in / rms_out)
    pk = float(np.max(np.abs(out)))
    if pk > 0.99:
        out = out * (0.99 / pk)
    return out


# ------------------------------------------------------------------ 基线自举
def base_measure(x, sr):
    """基线 f0 估计（无锚）：多配置 + 带内自洽。"""
    outs = []
    for kw in (dict(weight="inv_sqrt", norm_pow=0.5),
               dict(weight="inv", norm_pow=0.0),
               dict(weight="inv_sqrt", norm_pow=0.3)):
        f, _ = f0_anchored(x, sr, 200.0, band=(0.35, 3.2), per_frame=True, **kw)
        if f > 0:
            outs.append(f)
    try:
        fp, _, _ = librosa.pyin(x.astype(np.float32), fmin=45, fmax=800, sr=sr,
                                frame_length=2048, hop_length=512)
        v = fp[(fp > 45) & (fp < 800)]
        if len(v):
            outs.append(float(np.median(v)))
    except Exception:
        pass
    if not outs:
        return 0.0
    return float(np.median(outs))


# ------------------------------------------------------------------ 测试
def roundtrip():
    print("=" * 88)
    print("A. 基线锚定 round-trip（真实人声，已知变调倍率）")
    print("=" * 88)
    files = sorted(glob.glob(f"{VOICES}/samples/*_voice.wav")) + \
            sorted(glob.glob(f"{VOICES}/samples_ext/*_voice.wav"))
    step = max(1, len(files) // 40)
    files = files[::step][:40]
    facs = (0.5, 0.7, 0.85, 1.0, 1.3, 1.6, 2.0)
    res = {"anchored": [], "pyin": []}
    rows = []
    for p in files:
        cid = os.path.basename(p).split("_")[0]
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        b = base_measure(x, sr)
        if b <= 0:
            continue
        for fac in facs:
            y = x if abs(fac - 1.0) < 1e-6 else pitch_shift_once(x, sr, fac)
            # 锚定带以「变调后真值」为中心放宽，保证真值必在带内
            truth = b * fac
            band = (max(0.40, 0.75 * fac), min(3.0, 1.35 * fac))
            f, _ = f0_anchored_multi(y, sr, b, band=band)
            if f > 0:
                e = abs(f / truth - 1) * 100
                res["anchored"].append(e)
                rows.append((cid, b, fac, truth, f, e))
            try:
                fp, _, _ = librosa.pyin(y.astype(np.float32), fmin=45, fmax=1200,
                                        sr=sr, frame_length=2048, hop_length=512)
                v = fp[(fp > 45) & (fp < 1200)]
                fp = float(np.median(v)) if len(v) else 0.0
                if fp > 0:
                    res["pyin"].append(abs(fp / truth - 1) * 100)
            except Exception:
                pass
    print("%-16s %8s %8s %8s %8s" % ("估计器", "中位", "p75", "p90", "最大"))
    for name, arr in res.items():
        if not arr:
            continue
        a = np.array(arr)
        print("%-16s %7.2f%% %7.2f%% %7.2f%% %8.2f%%  (n=%d)"
              % (name, np.median(a), np.percentile(a, 75),
                 np.percentile(a, 90), np.max(a), len(a)))
    rows.sort(key=lambda r: -r[5])
    print("\n--- 锚定法误差最大的 12 条 ---")
    for r in rows[:12]:
        print("  %-6s base=%6.1f x%.2f  真值=%7.1f  读数=%7.1f  误差=%6.2f%%"
              % (r[0], r[1], r[2], r[3], r[4], r[5]))
    return res


def case_check():
    print()
    print("=" * 88)
    print("B. 失败案例：锚定读数 vs pyin 读数（带宽收窄到 0.55~2.30）")
    print("=" * 88)
    base = json.load(open(f"{VOICES}/base_f0_v10.json"))
    cases = [("M28", "nu"), ("M02", "nu"), ("M52", "yan"), ("M65", "nu"),
             ("M11", "yan"), ("M71", "nu"), ("M77", "nu"), ("M09", "nu"),
             ("M13", "yan"), ("M25", "yan"), ("M28", "xi"), ("M02", "yan")]
    print("%-10s %7s %8s %8s %8s %8s" %
          ("case", "base", "pyin", "pyin/b", "anchor", "anchor/b"))
    for cid, emo in cases:
        ep = glob.glob(f"{EMOTIONS}/{cid}/*_{emo}.wav")
        if not ep:
            continue
        x, sr = sf.read(ep[0], dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        b = base.get(cid, 0)
        try:
            fp, _, _ = librosa.pyin(x.astype(np.float32), fmin=40, fmax=1200,
                                    sr=sr, frame_length=2048, hop_length=512)
            v = fp[(fp > 40) & (fp < 1200)]
            pv = float(np.median(v)) if len(v) else 0.0
        except Exception:
            pv = 0.0
        fa, det = f0_anchored_multi(x, sr, b)
        print("%-10s %7.1f %8.1f %8.2f %8.1f %8.2f   %s"
              % (f"{cid}_{emo}", b, pv, pv / b if b else 0,
                 fa, fa / b if b else 0, det.get("vals")))


if __name__ == "__main__":
    roundtrip()
    case_check()
