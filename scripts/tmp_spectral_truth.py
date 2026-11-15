"""决定性诊断：真实频谱里，失败案例的 f0 到底是多少？

做法：不依赖任何 f0 估计器，直接看长窗频谱的峰列。
  - 若真 f0 = 74Hz，则谐波峰应出现在 74/148/222/296/370/444/...（等间距 74）
  - 若真 f0 = 740Hz，则峰出现在 740/1480/...
  - 峰间距直接给出 f0，不存在倍频歧义。

对比：基线样本 vs 情绪样本的峰列，判断「情绪音高真的跑飞」还是「测量锁谐波」。
"""
import glob
import json
import os

import librosa
import numpy as np
import soundfile as sf

VOICES = "/ssd2/duanju/voices"
EMOTIONS = f"{VOICES}/emotions"


def peak_list(x, sr, n_fft=None, fmax=1600, top=14):
    """整段 FFT 的显著峰（Hz），频率分辨率 0.2Hz 级。"""
    if n_fft is None:
        n_fft = 1 << int(np.ceil(np.log2(max(len(x), 4096))))
    w = np.hanning(len(x)).astype(np.float32)
    X = np.abs(np.fft.rfft(x * w, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    S = X.copy()
    # 轻度平滑
    k = np.hanning(9)
    k /= k.sum()
    S = np.convolve(S, k, mode="same")
    lim = np.searchsorted(freqs, fmax)
    idx = []
    S2 = S[:lim].copy()
    for _ in range(top):
        i = int(np.argmax(S2))
        if S2[i] <= 0:
            break
        # 抛物线细化
        if 0 < i < len(S) - 1:
            y0, y1, y2 = S[i - 1], S[i], S[i + 1]
            den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
            if abs(d) > 1:
                d = 0.0
            f = freqs[i] + d * (freqs[1] - freqs[0])
        else:
            f = freqs[i]
        idx.append((float(f), float(S[i])))
        lo = max(0, i - 25)
        hi = min(len(S2), i + 25)
        S2[lo:hi] = 0
    return idx


def spacing_f0(peaks, tol=0.08):
    """从峰列推断基频：取所有峰差值的「公共间距」候选。"""
    fs = [p[0] for p in peaks if p[0] > 40]
    if len(fs) < 3:
        return 0.0
    diffs = []
    for i in range(len(fs) - 1):
        for j in range(i + 1, len(fs)):
            d = fs[j] - fs[i]
            if 40 < d < 500:
                diffs.append(d)
    if not diffs:
        return 0.0
    # 聚类
    diffs = np.array(diffs)
    best, bestn = 0.0, 0
    for d in diffs:
        n = int(np.sum(np.abs(diffs - d) / d < tol))
        if n > bestn:
            bestn, best = n, d
    return float(best)


def show(cid, emo):
    bp = glob.glob(f"{VOICES}/samples/{cid}_*_voice.wav") + \
         glob.glob(f"{VOICES}/samples_ext/{cid}_*_voice.wav")
    ep = glob.glob(f"{EMOTIONS}/{cid}/*_{emo}.wav")
    if not bp or not ep:
        print(f"  {cid} {emo}: 文件缺失")
        return
    bx, sr = sf.read(bp[0], dtype="float32")
    if bx.ndim > 1:
        bx = bx[:, 0]
    ex, _ = sf.read(ep[0], dtype="float32")
    if ex.ndim > 1:
        ex = ex[:, 0]

    bp_ = peak_list(bx, sr)
    ep_ = peak_list(ex, sr)
    bsp = spacing_f0(bp_)
    esp = spacing_f0(ep_)
    print(f"  {cid}_{emo}")
    print(f"    基线峰列: " + " ".join(f"{f:.0f}" for f, _ in bp_[:10]))
    print(f"    基线间距推断 f0 = {bsp:.1f}")
    print(f"    情绪峰列: " + " ".join(f"{f:.0f}" for f, _ in ep_[:10]))
    print(f"    情绪间距推断 f0 = {esp:.1f}")
    # 各估计器读数
    try:
        f, _, _ = librosa.pyin(ex.astype(np.float32), fmin=45, fmax=800, sr=sr,
                               frame_length=2048, hop_length=512)
        v = f[(f > 45) & (f < 800)]
        print(f"    pyin 读数 = {float(np.median(v)) if len(v) else 0:.1f}")
    except Exception as e:
        print("    pyin err", e)
    print()


if __name__ == "__main__":
    base = json.load(open(f"{VOICES}/base_f0_v10.json"))
    cases = [("M02", "yan"), ("M52", "yan"), ("M65", "nu"), ("M28", "nu"),
             ("M13", "yan"), ("M25", "yan"), ("M09", "nu"), ("M11", "yan"),
             ("M71", "nu"), ("M77", "nu"), ("M02", "nu"), ("M28", "xi")]
    print("=" * 84)
    print("真实频谱诊断：失败案例的基线 vs 情绪")
    print("=" * 84)
    for cid, emo in cases:
        show(cid, emo)
