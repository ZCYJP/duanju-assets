"""帧级 + 谐波存在性的最终 f0 判定。

问题回顾
--------
单条长窗 FFT 的峰列不可靠（语调漂移会把谐波峰展宽成簇）；
pyin 帧级中位数偶尔锁到高阶谐波（M01 实测：真 f0≈90，pyin 读 112，
而长窗峰列显示 190/194 处能量最强 —— pyin 锁的是 ~190 的谐波）。

本方法
------
1. **帧级 pyin** 拿完整 f0 轨迹（含所有帧的读数，不只中位数）
2. 对每个候选 f（来自轨迹的密集簇），做**谐波存在性检验**：
   把音频重采样到 4×f0 的采样率（变精确到 1/4 周期），
   然后看 1/f0 周期处是否有稳定的峰（基频存在性）。
   这等价于「用 4 倍频的强谐波去定位基频」，
   而高阶谐波在 f0/2、f0/4 处是静音。

   实现上：用「梳状滤波匹配」在候选 f0 的 1/2、1/4 处取相位一致性。
   更简单可靠的等价做法：
   **自相关峰间距检验** —— 对候选 f，计算 R(f) 与 R(f/2)，
   若 R(f/2) 也很高，说明 f/2 才是周期。

   最稳妥：用 librosa 的 yin's difference function 自己取**全局最小**
   （不做 first-dip 规则），再与 first-dip 结果对比：
     - first-dip 通常是真实基频（抗倍频但可能受噪声影响）
     - global-min 通常是锁到高阶
   若两者相差 2 倍，取 first-dip。

3. 取「帧级 first-dip 众数」作为最终 f0，再用帧级分布的
   **连续性**（变化平滑性）验证：真实 f0 轨迹应连续，
   锁到高阶谐波的轨迹会呈现「整数倍跳变」。

验证：合成信号（已知真值）+ 真实 round-trip（已知倍率）。
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import soundfile as sf

VOICES = "/ssd2/duanju/voices"
CAST = "/ssd2/duanju/scripts/cast_all_150.json"

GENDER_RANGE = {"男": (75.0, 200.0), "女": (150.0, 285.0)}
HARD_LO, HARD_HI = 55.0, 500.0


def frame_pyin(x, sr):
    """帧级 pyin 全轨迹。"""
    try:
        f, _, _ = librosa.pyin(x.astype(np.float32), fmin=40.0, fmax=800.0, sr=sr,
                               frame_length=2048, hop_length=256)
        v = f[np.isfinite(f)]
        return v
    except Exception:
        return np.array([])


def yin_cmnd_track(x, sr, fmin=45.0, fmax=600.0, frame_len=2048, hop=512,
                   thresh=0.15):
    """帧级 YIN（first-dip 规则），返回每帧 f0。

    实现：对每帧单独算差分函数的归一化版本。
    为控制成本，帧数上限 300。
    """
    n = len(x)
    if n < frame_len * 2:
        return np.array([])
    starts = list(range(0, n - frame_len, hop))
    if len(starts) > 300:
        idx = np.linspace(0, len(starts) - 1, 300).astype(int)
        starts = [starts[i] for i in idx]
    tau_min = max(2, int(sr / fmax))
    tau_max = int(sr / fmin)
    if tau_max + 8 >= frame_len:
        tau_max = frame_len - 8
    out = []
    for s in starts:
        seg = x[s:s + frame_len].astype(np.float64)
        seg = seg - seg.mean()
        W = frame_len
        nfft = 1 << int(np.ceil(np.log2(2 * W)))
        A = np.fft.rfft(seg, nfft)
        R = np.fft.irfft(A * np.conj(A), nfft)[:tau_max + 1]
        cum = np.concatenate([[0.0], np.cumsum(seg * seg)])
        E1 = cum[W]
        E2 = cum[W] - cum[:tau_max + 1]
        e = np.maximum(E1 + E2 - 2.0 * R, 0.0)
        e[:tau_min] = np.inf
        fin = np.isfinite(e)
        safe = np.where(fin, e, 0.0)
        csum = np.cumsum(safe)
        cnt = np.cumsum(fin.astype(np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            cmnd = np.where(fin & (cnt > 0),
                            e * np.arange(len(e)) / np.maximum(csum, 1e-12), 1.0)
        seg_cm = cmnd[tau_min:tau_max + 1]
        if not np.any(np.isfinite(seg_cm)):
            continue
        cand = None
        for i in range(1, len(seg_cm) - 1):
            v = seg_cm[i]
            if np.isfinite(v) and v < thresh and v <= seg_cm[i - 1] and v < seg_cm[i + 1]:
                cand = i + tau_min
                break
        if cand is None:
            continue
        if tau_min < cand < tau_max:
            y0, y1, y2 = cmnd[cand - 1], cmnd[cand], cmnd[cand + 1]
            den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
            if abs(d) > 1.0:
                d = 0.0
            tau = cand + d
        else:
            tau = cand
        out.append(sr / max(tau, 1e-9))
    return np.array(out)


def octave_correction(f_track):
    """用轨迹的「连续性」纠正八度跳变。

    真实 f0 轨迹在相邻帧间变化平缓（<20%）。
    若某帧读数是邻域中位数的 2 倍或 0.5 倍且邻域一致，
    则该帧锁错八度 —— 但这只能纠正**帧内**，不能纠正**整体**。

    因此本函数返回：
      - raw_median   原始中位数
      - continuity_ok 轨迹平滑性（相邻帧比值的中位绝对偏差）
      - octave_span  轨迹中读数的「跨度倍数」分布
    """
    if len(f_track) < 5:
        return {"raw_median": float(np.median(f_track)) if len(f_track) else 0.0,
                "continuity": 0.0, "span": 0.0}
    r = f_track[1:] / np.maximum(f_track[:-1], 1e-9)
    good = r[(r > 0.4) & (r < 2.5)]
    cont = float(np.median(np.abs(np.log2(np.maximum(good, 1e-9))))) if len(good) else 0.0
    # 跨度：把读数按 2 的幂聚类，看有多少个八度簇
    lg = np.log2(np.maximum(f_track, 1e-9))
    base = np.median(lg)
    oct_idx = np.round(lg - base).astype(int)
    vals, cnts = np.unique(oct_idx, return_counts=True)
    span = float(cnts.max()) / len(oct_idx)
    return {"raw_median": round(float(np.median(f_track)), 2),
            "continuity": round(cont, 4),
            "span": round(span, 3)}


def harmonic_check(x, sr, f):
    """谐波存在性：f 处与其 1/2、1/4 处的谱峰能量比。

    做法：长窗频谱 -> 在 f, f/2, f/4 处各取局部峰能量（归一到全局峰）。
    若 f/2 或 f/4 也有相当能量，则真 f0 更低。
    注意：此判据在高 f0（>250Hz）时 f/2 落入共振峰区会误判，
    故只作为辅助信号，权重低。
    """
    n_fft = 1 << int(np.ceil(np.log2(max(len(x), 16384))))
    w = np.hanning(len(x)).astype(np.float32)
    S = np.abs(np.fft.rfft(x * w, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    df = float(freqs[1] - freqs[0])
    lim = int(np.searchsorted(freqs, 3000.0))
    gmax = float(S[:lim].max())
    if gmax <= 0:
        return {"e_f": 0, "e_half": 0, "e_quarter": 0, "ok": True}

    def at(fr):
        if fr < df * 2 or fr > freqs[lim - 1]:
            return 0.0
        i0 = int(fr / df)
        wd = max(2, int(0.03 * fr / df))
        lo = max(1, i0 - wd)
        hi = min(lim - 1, i0 + wd)
        j = lo + int(np.argmax(S[lo:hi]))
        return float(S[j]) / gmax

    e_f, e_h, e_q = at(f), at(f / 2.0), at(f / 4.0)
    # f/2 有实质能量 => 可能锁高八度
    lock_high = (e_h >= 0.30 * e_f and e_h > 0.03)
    lock_double = (e_q >= 0.30 * e_f and e_q > 0.02 and e_h < e_f * 0.15)
    return {"e_f": round(e_f, 4), "e_half": round(e_h, 4),
            "e_quarter": round(e_q, 4),
            "lock_high": bool(lock_high), "lock_double": bool(lock_double)}


def analyze(x, sr, gender):
    ft = frame_pyin(x, sr)
    yt = yin_cmnd_track(x, sr)
    info = {}
    if len(ft):
        info.update({f"pyin_{k}": v for k, v in octave_correction(ft).items()})
    if len(yt):
        info.update({f"yin_{k}": v for k, v in octave_correction(yt).items()})

    # 候选集合：pyin 众数、yin 众数、各自的 1/2 与 2 倍
    cands = set()
    if len(ft):
        cands.add(round(float(np.median(ft)), 2))
    if len(yt):
        cands.add(round(float(np.median(yt)), 2))
    for c in list(cands):
        for m in (0.5, 2.0):
            v = c * m
            if HARD_LO <= v <= HARD_HI:
                cands.add(round(v, 2))

    lo, hi = GENDER_RANGE.get(gender, (60.0, 300.0))
    # 逐候选评分：帧级一致性（有多少帧支持）+ 谐波存在性
    best = None
    for c in sorted(cands):
        if c <= 0:
            continue
        # 支持度：pyin 轨迹中与 c 相差 <=8% 的帧占比
        sup_py = float(np.mean(np.abs(ft / c - 1) <= 0.08)) if len(ft) else 0.0
        sup_yin = float(np.mean(np.abs(yt / c - 1) <= 0.08)) if len(yt) else 0.0
        hc = harmonic_check(x, sr, c)
        penalty = 0.0
        if hc["lock_high"]:
            penalty += 0.55
        if hc["lock_double"]:
            penalty += 0.45
        inr = 1.0 if (lo <= c <= hi) else 0.0
        score = (sup_py + sup_yin) * 0.5 - penalty - inr * 0.30
        if best is None or score > best[0]:
            best = (score, c, sup_py, sup_yin, hc, inr)
    if best is None:
        return 0.0, info
    _, c, sup_py, sup_yin, hc, inr = best
    info.update({"f0_final": c, "sup_pyin": round(sup_py, 3),
                  "sup_yin": round(sup_yin, 3), "in_range": inr == 0.0,
                  **hc})
    return round(float(c), 2), info


def one(c):
    cid = c["id"]
    g = c.get("gender", "")
    p = ""
    for d in ("samples", "samples_ext"):
        g2 = sorted(glob.glob(f"{VOICES}/{d}/{cid}_*_voice.wav"))
        if g2:
            p = g2[0]
            break
    if not p:
        return None
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    f, info = analyze(x, sr, g)
    return {"cid": cid, "gender": g, "f0": f, "path": p, **info}


def main():
    cast = json.load(open(CAST))
    print(f"[1/3] 帧级 + 谐波存在性分析 {len(cast)} 个基线音色")
    with ProcessPoolExecutor(max_workers=48) as ex:
        rows = list(ex.map(one, cast, chunksize=1))
    rows = [r for r in rows if r]

    print("[2/3] 统计")
    for g in ("男", "女"):
        grp = [r for r in rows if r["gender"] == g]
        v = np.array([r["f0"] for r in grp])
        lo, hi = GENDER_RANGE[g]
        print(f"  {g}: n={len(v)} 中位 {np.median(v):.1f}  "
              f"区间 {v.min():.1f}~{v.max():.1f}  落音域 "
              f"{int(((v>=lo)&(v<=hi)).sum())}/{len(v)}")
        for r in grp:
            if not (lo <= r["f0"] <= hi):
                print(f"     OUT {r['cid']} f0={r['f0']:.1f} "
                      f"pyin_med={r.get('pyin_raw_median')} "
                      f"yin_med={r.get('yin_raw_median')} "
                      f"sup_py={r.get('sup_pyin')} sup_yin={r.get('sup_yin')} "
                      f"e_f={r.get('e_f')} e_h={r.get('e_half')}")

    print("[3/3] 写出")
    json.dump(rows, open(f"{VOICES}/base_f0_frame.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"  -> {VOICES}/base_f0_frame.json")


if __name__ == "__main__":
    main()
