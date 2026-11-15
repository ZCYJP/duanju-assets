"""f0 估计器实验室：谐波梳状谱（harmonic comb）八度消歧。

问题
----
所有时域估计器（pyin / yin_thr / ac）在极低音区会锁到 2f、3f 甚至更高次谐波：
  - f0=74Hz 时，2f=148Hz 的谐波能量远高于基频，估计器无法分辨
  - 于是「测量值」可能是真值的 2 倍、3 倍、10 倍
这导致回正时把 factor 算错（差一个整数倍），永远调不到位。

思路
----
谐波梳状谱（harmonic summation / comb filter）：
  对候选基频 f，score(f) = Σ_h w_h · S(h·f)
  真基频处所有谐波都对齐，能量最全；
  高次谐波候选（2f 真值）只有 1~2 个谐波落在真实能量上，score 低。
  归一化（除以谐波数）可消除「低 f 收集更多谐波」的天然偏置。

配套：同一算法同时用于基线与情绪音频，保证比值口径一致。

输出
----
1) 合成信号精度（已知真值）
2) 真实基线音频的 round-trip 精度（已知变调倍率）
3) 择优后的参数配置
"""
import glob
import json
import os
import sys

import librosa
import numpy as np
import soundfile as sf

VOICES = "/ssd2/duanju/voices"
OUT = "/ssd2/duanju/voices/f0_comb_lab.json"


# ------------------------------------------------------------------ 谱
def median_spectrum(x, sr, n_fft=16384, hop=4096):
    """多帧中值幅度谱（比单帧稳，抑制个别帧的谐波抖动）。"""
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    frames = librosa.util.frame(x, frame_length=n_fft, hop_length=hop)
    w = np.hanning(n_fft).astype(np.float32)
    # librosa.util.frame 对 1D 输入返回 (frame_length, n_frames)
    S = np.abs(np.fft.rfft(frames * w[:, None], axis=0))  # (bins, frames)
    return np.median(S, axis=1), np.fft.rfftfreq(n_fft, 1.0 / sr)


# ------------------------------------------------------------------ 变调
def pitch_shift_once(x, sr, factor):
    """与已验证的变调算子一致（精度 0.1~1.1%）。"""
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


# ------------------------------------------------------------------ 梳状谱
def comb_scores(x, sr, grid, comb_max, weight, norm_pow, sqrt_amp):
    """对每个候选 f 计算谐波梳得分。"""
    S, freqs = median_spectrum(x, sr)
    if sqrt_amp:
        S = np.sqrt(S)
    H = np.maximum((comb_max / grid).astype(int), 1)
    hmax = int(H.max())
    score = np.zeros_like(grid)
    for h in range(1, hmax + 1):
        mask = H >= h
        if not np.any(mask):
            continue
        f_h = grid[mask] * h
        amp = np.interp(f_h, freqs, S, left=0.0, right=0.0)
        score[mask] += weight(h) * amp
    denom = np.power(H.astype(np.float64), norm_pow)
    return score / np.maximum(denom, 1e-9)


def f0_comb(x, sr, fmin=45.0, fmax=650.0, df=0.5, comb_max=2200.0,
            weight="inv_sqrt", norm_pow=0.5, sqrt_amp=True, refine=True):
    """返回梳状谱估计的 f0。"""
    grid = np.arange(fmin, fmax + df, df)
    if weight == "inv":
        w = lambda h: 1.0 / h
    elif weight == "inv_sqrt":
        w = lambda h: 1.0 / np.sqrt(h)
    else:
        w = lambda h: 1.0
    score = comb_scores(x, sr, grid, comb_max, w, norm_pow, sqrt_amp)
    i = int(np.argmax(score))
    if not refine or i <= 0 or i >= len(grid) - 1:
        return float(grid[i]), float(score[i])
    # 抛物线细化
    y0, y1, y2 = score[i - 1], score[i], score[i + 1]
    den = y0 - 2 * y1 + y2
    d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
    if abs(d) > 1.0:
        d = 0.0
    return float(grid[i] + d * df), float(y1)


# ------------------------------------------------------------------ 基线候选
def base_anchor_candidates(x, sr, grid_df=0.5):
    """给出多种八度候选：梳状谱及其 2^k 降八度。

    消歧原则：情绪音频与角色基线同源，音高应在同量级（0.45~2.2x）。
    因此把测量值的降八度候选全部列出，由后续逻辑挑选。
    """
    cands = []
    for kw in (
        dict(norm_pow=0.5, sqrt_amp=True, weight="inv_sqrt"),
        dict(norm_pow=0.0, sqrt_amp=False, weight="inv"),
        dict(norm_pow=1.0, sqrt_amp=True, weight="inv_sqrt"),
        dict(norm_pow=0.5, sqrt_amp=False, weight="inv"),
    ):
        f, _ = f0_comb(x, sr, df=grid_df, **kw)
        if f > 0:
            for k in range(0, 5):
                cands.append(f / (2 ** k))
    return cands


# ------------------------------------------------------------------ 合成测试
def synth_test():
    print("=" * 78)
    print("A. 合成信号精度（谐波梳状谱）")
    print("=" * 78)
    sr = 24000
    rows = []
    rng = np.random.default_rng(7)
    for f_true in (60.0, 70.0, 74.0, 92.0, 110.0, 150.0, 200.0, 300.0, 420.0):
        t = np.arange(int(sr * 2.0)) / sr
        # 声源：脉冲串 + 若干谐波 + 噪声
        x = np.zeros_like(t)
        for h in range(1, 13):
            x += (1.0 / h) * np.sin(2 * np.pi * f_true * h * t +
                                    rng.uniform(0, 0.3))
        # 让 2f、3f 能量更高（模拟真实人声的倍频误锁场景）
        x += 0.9 * np.sin(2 * np.pi * f_true * 2 * t)
        x += 0.6 * np.sin(2 * np.pi * f_true * 3 * t)
        x += 0.02 * rng.standard_normal(len(t))
        x = (x / np.max(np.abs(x)) * 0.6).astype(np.float32)
        best = None
        for kw in (dict(norm_pow=0.5, sqrt_amp=True, weight="inv_sqrt"),
                   dict(norm_pow=0.0, sqrt_amp=False, weight="inv"),
                   dict(norm_pow=1.0, sqrt_amp=True, weight="inv_sqrt"),
                   dict(norm_pow=0.5, sqrt_amp=False, weight="inv")):
            f, _ = f0_comb(x, sr, **kw)
            err = abs(f - f_true) / f_true * 100
            if best is None or err < best[1]:
                best = (f, err, kw)
        rows.append((f_true, best))
        print("  f0=%6.1f  -> %7.2f Hz  误差 %5.2f%%   cfg=%s"
              % (f_true, best[0], best[1],
                 {k: v for k, v in best[2].items()}))
    errs = [r[1][1] for r in rows]
    print("  合成测试: 中位误差 %.2f%%  最大 %.2f%%"
          % (np.median(errs), np.max(errs)))
    return rows


# ------------------------------------------------------------------ 真实 round-trip
def roundtrip_test(factors=(0.5, 0.7, 0.85, 1.0, 1.3, 1.6, 2.0)):
    print()
    print("=" * 78)
    print("B. 真实音频 round-trip（已知变调倍率）")
    print("=" * 78)
    files = sorted(glob.glob(f"{VOICES}/samples/*_voice.wav")) + \
            sorted(glob.glob(f"{VOICES}/samples_ext/*_voice.wav"))
    # 取 30 条覆盖各音域
    step = max(1, len(files) // 30)
    files = files[::step][:30]

    configs = {
        "comb_np05_sqrt": dict(norm_pow=0.5, sqrt_amp=True, weight="inv_sqrt"),
        "comb_np0_raw": dict(norm_pow=0.0, sqrt_amp=False, weight="inv"),
        "comb_np1_sqrt": dict(norm_pow=1.0, sqrt_amp=True, weight="inv_sqrt"),
        "comb_np05_raw": dict(norm_pow=0.5, sqrt_amp=False, weight="inv"),
    }
    stat = {k: [] for k in configs}
    stat["pyin"] = []
    for p in files:
        cid = os.path.basename(p).split("_")[0]
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        for fac in factors:
            y = x if abs(fac - 1.0) < 1e-6 else pitch_shift_once(x, sr, fac)
            # 真值 = 原音频真 f0 * fac；原音频真 f0 未知，改用「相对原测量值」
            # 这里用梳状谱在原音频上的值作为参考真值（先验：原音频在其可靠区）
            for name, kw in configs.items():
                f, _ = f0_comb(y, sr, **kw)
                f0, _ = f0_comb(x, sr, **kw)
                if f0 > 0:
                    stat[name].append(abs(f / f0 - fac) / fac * 100)
            try:
                fp, _, _ = librosa.pyin(y.astype(np.float32), fmin=45, fmax=800,
                                        sr=sr, frame_length=2048, hop_length=512)
                vp = fp[(fp > 45) & (fp < 800)]
                fp = float(np.median(vp)) if len(vp) else 0.0
                f0p, _, _ = librosa.pyin(x.astype(np.float32), fmin=45, fmax=800,
                                         sr=sr, frame_length=2048, hop_length=512)
                v0 = f0p[(f0p > 45) & (f0p < 800)]
                f0p = float(np.median(v0)) if len(v0) else 0.0
                if f0p > 0 and fp > 0:
                    stat["pyin"].append(abs(fp / f0p - fac) / fac * 100)
            except Exception:
                pass
    print("%-18s %8s %8s %8s %8s" % ("估计器", "中位", "p75", "p90", "最大"))
    out = {}
    for name, arr in stat.items():
        if not arr:
            continue
        a = np.array(arr)
        out[name] = {"median": float(np.median(a)),
                     "p75": float(np.percentile(a, 75)),
                     "p90": float(np.percentile(a, 90)),
                     "max": float(np.max(a)),
                     "n": len(a)}
        print("%-18s %7.2f%% %7.2f%% %7.2f%% %7.2f%%  (n=%d)"
              % (name, out[name]["median"], out[name]["p75"],
                 out[name]["p90"], out[name]["max"], len(a)))
    return out


# ------------------------------------------------------------------ 八度稳健性
def octave_test():
    """关键测试：对真实音频做「降八度」变调，估计器应仍报出降八度后的值。"""
    print()
    print("=" * 78)
    print("C. 八度稳健性（对真实人声降八度后能否正确跟踪）")
    print("=" * 78)
    files = sorted(glob.glob(f"{VOICES}/samples/*_voice.wav")) + \
            sorted(glob.glob(f"{VOICES}/samples_ext/*_voice.wav"))
    # 挑基线 f0 较高的（v10 口径），降八度后落在 70~180Hz 危险区
    base = json.load(open(f"{VOICES}/base_f0_v10.json"))
    picks = []
    for p in files:
        cid = os.path.basename(p).split("_")[0]
        f = base.get(cid, 0)
        if 150 <= f <= 320:
            picks.append((p, f))
    picks = picks[:14]
    configs = {
        "comb_np05_sqrt": dict(norm_pow=0.5, sqrt_amp=True, weight="inv_sqrt"),
        "comb_np0_raw": dict(norm_pow=0.0, sqrt_amp=False, weight="inv"),
        "comb_np1_sqrt": dict(norm_pow=1.0, sqrt_amp=True, weight="inv_sqrt"),
    }
    for name, kw in configs.items():
        errs = []
        for p, f0 in picks:
            x, sr = sf.read(p, dtype="float32")
            if x.ndim > 1:
                x = x[:, 0]
            y = pitch_shift_once(x, sr, 0.5)
            f, _ = f0_comb(y, sr, **kw)
            if f > 0:
                errs.append(abs(f / (f0 * 0.5) - 1) * 100)
        if errs:
            a = np.array(errs)
            print("  %-16s n=%2d 中位 %6.2f%%  p90 %6.2f%%  最大 %7.2f%%"
                  % (name, len(a), np.median(a), np.percentile(a, 90), np.max(a)))
    try:
        errs = []
        for p, f0 in picks:
            x, sr = sf.read(p, dtype="float32")
            if x.ndim > 1:
                x = x[:, 0]
            y = pitch_shift_once(x, sr, 0.5)
            fp, _, _ = librosa.pyin(y.astype(np.float32), fmin=45, fmax=800,
                                    sr=sr, frame_length=2048, hop_length=512)
            v = fp[(fp > 45) & (fp < 800)]
            fp = float(np.median(v)) if len(v) else 0.0
            if fp > 0:
                errs.append(abs(fp / (f0 * 0.5) - 1) * 100)
        if errs:
            a = np.array(errs)
            print("  %-16s n=%2d 中位 %6.2f%%  p90 %6.2f%%  最大 %7.2f%%"
                  % ("pyin", len(a), np.median(a), np.percentile(a, 90), np.max(a)))
    except Exception as e:
        print("  pyin err", e)


if __name__ == "__main__":
    res = {"synth": [], "roundtrip": {}, "octave": {}}
    s = synth_test()
    res["synth"] = [{"f_true": r[0], "f_est": r[1][0], "err": r[1][1]} for r in s]
    res["roundtrip"] = roundtrip_test()
    octave_test()
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)
    print("\n->", OUT)
