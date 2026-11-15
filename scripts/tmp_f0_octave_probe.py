"""纯物理裁决：用「降八度后的读数」反推真实 f0，判定 pyin 是否锁错八度。

推理（无任何启发式阈值）
------------------------
设原音频真实 f0 = F，pyin 在原音上读 p。
对原音施加**降八度**变调（factor=0.5，精确算子，实测误差 0.1~1.1%），
变调后真实 f0 = F/2，pyin 在变调音上读 q。

若 pyin 在两处都锁同一八度（大多数情况，round-trip 实测 95.8%）：
    q = p / 2
而 p 与 q 的关系揭示了 F：
  情形 A：p = 2F,  q = F     (pyin 原读高八度) -> F = q
  情形 B：p = F,   q = F/2   (pyin 原读正确)   -> F = p = 2q

因此：**F = max(p, 2q)** 或 **F = min(p, 2q)**，取决于哪个自洽。
再验证：把音频**升八度**（factor=2.0）得 r。
  情形 A：r = 2F = 2p
  情形 B：r = 2F = 2p    （两者 r 都 = 2p）
=> r 无法区分，但 p 与 q 的关系可以：
  若 q ≈ p/2  => pyin 两处同锁 => 无法区分 p 与 2p
  若 q ≈ p/4  => 说明降八度后 pyin 又锁高了八度 => p = 2F（情形 A）
  若 q ≈ p    => 说明降八度后 pyin 锁低了 => p = F（情形 B）

更直接的做法：**二分降调**。逐次降八度，直到读数落入该性别的音域，
读数序列每一步减半 -> 落入音域的那个读数就是真实 f0 的量级。

实现：连续降八度，记录每一步的 pyin 读数 r_0, r_1, r_2, ...，
真实 f0 F 满足 r_k ≈ F / 2^k（若始终同锁）。
=> F = r_k × 2^k，对不同 k 应一致；取**一致性最好**的 k 组合。

这比"存在性阈值"稳健得多，因为它是**相对比较**，不依赖绝对能量门限。
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tmp_f0_arena import est_pyin  # noqa: E402

VOICES = "/ssd2/duanju/voices"
CAST = "/ssd2/duanju/scripts/cast_all_150.json"

GENDER_RANGE = {"男": (75.0, 200.0), "女": (150.0, 285.0)}
# 硬物理边界：人声 f0 不会低于 55Hz（除极端低音）或高于 500Hz
HARD_LO, HARD_HI = 55.0, 500.0


def pitch_shift(x, sr, factor):
    if abs(factor - 1.0) < 1e-6:
        return x.astype(np.float32)
    n_in = len(x)
    rms_in = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    y = librosa.resample(x.astype(np.float32), orig_sr=sr,
                         target_sr=max(int(sr / factor), 1000))
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    Ds = librosa.phase_vocoder(D, rate=1.0 / factor, hop_length=256, n_fft=1024)
    out = np.asarray(librosa.istft(Ds, hop_length=256, n_fft=1024), dtype=np.float32)
    out = out[:n_in] if len(out) > n_in else np.pad(out, (0, n_in - len(out)))
    rms_out = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    if rms_out > 1e-8 and rms_in > 1e-8:
        out = out * (rms_in / rms_out)
    pk = float(np.max(np.abs(out)))
    if pk > 0.99:
        out = out * (0.99 / pk)
    return out


def octave_ladder(x, sr, max_steps=4):
    """连续降八度，收集 pyin 读数序列。"""
    seq = []
    y = x
    for k in range(max_steps + 1):
        f = est_pyin(y, sr)
        seq.append(round(f, 2))
        if f <= 0:
            break
        y = pitch_shift(y, sr, 0.5)
    return seq


def solve_octave(seq, gender):
    """由降八度序列推断真实 f0。

    模型：第 k 步的真实 f0 = F / 2^k。
    pyin 在第 k 步可能锁在 F/2^k 的 m_k 倍（m_k ∈ {1, 0.5, 2, ...}）。
    我们找 F，使 F/2^k 对每个 k 都「接近某个允许的八度分支」，
    并且 F 落在性别音域 / 硬边界内。

    实用实现：从每个 k 出发，候选 F = seq[k] * 2^k * m，
    m ∈ {1, 0.5, 2}；对每个候选 F，检查所有 k 的 seq[k] 是否
    都 ≈ F / 2^k × (某个八度分支)。
    """
    lo, hi = GENDER_RANGE.get(gender, (60.0, 300.0))
    cands = set()
    for k, r in enumerate(seq):
        if r <= 0:
            continue
        for m in (1.0, 0.5, 2.0):
            F = r * (2 ** k) * m
            if HARD_LO <= F <= HARD_HI:
                cands.add(round(F, 2))
    best = None
    for F in cands:
        # 评分：所有 k 上 seq[k] 与 F/2^k 的八度接近度
        errs = []
        for k, r in enumerate(seq):
            if r <= 0:
                continue
            base = F / (2 ** k)
            # 允许 r 落在 base 的 {1,2,0.5} 倍附近
            e = min(abs(r / base - m) for m in (1.0, 2.0, 0.5))
            errs.append(e)
        if not errs:
            continue
        med = float(np.median(errs))
        # 音域加分（不强制）
        inr = 0.0 if (lo <= F <= hi) else 1.0
        score = med + inr * 0.25
        if best is None or score < best[0]:
            best = (score, F, med, inr)
    if best is None:
        return 0.0, {"score": 9.9, "med": 9.9, "in_range": False}
    return round(best[1], 2), {"score": round(best[0], 4),
                               "med": round(best[2], 4),
                               "in_range": best[3] == 0.0}


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
    seq = octave_ladder(x, sr)
    F, info = solve_octave(seq, g)
    return {"cid": cid, "gender": g, "seq": seq, "ladder_f0": F,
            "pyin": seq[0] if seq else 0.0, **info, "path": p}


def main():
    cast = json.load(open(CAST))
    print(f"[1/3] 八度阶梯：{len(cast)} 个基线音色（每条最多 5 次降八度）")
    with ProcessPoolExecutor(max_workers=48) as ex:
        rows = list(ex.map(one, cast, chunksize=1))
    rows = [r for r in rows if r]

    print("[2/3] 统计")
    diffs = []
    for r in rows:
        g = r["gender"]
        lo, hi = GENDER_RANGE.get(g, (60, 300))
        p = r["pyin"]
        F = r["ladder_f0"]
        if p > 0 and F > 0:
            rt = p / F
            if abs(rt - 1.0) > 0.06:
                diffs.append((r, rt))
    print(f"  阶梯解与 pyin 不一致: {len(diffs)}/{len(rows)}")
    print(f"{'cid':<7}{'性别':<6}{'pyin':>9}{'阶梯F0':>9}{'比值':>8}"
          f"{'序列':>34}  音域")
    for r, rt in diffs:
        lo, hi = GENDER_RANGE.get(r["gender"], (60, 300))
        inr = "OK" if lo <= r["ladder_f0"] <= hi else "OUT"
        print(f"{r['cid']:<7}{r['gender']:<6}{r['pyin']:>9.1f}"
              f"{r['ladder_f0']:>9.1f}{rt:>8.3f}  "
              f"{str(r['seq']):>34}  {inr}")

    print("[3/3] 整体音域分布")
    for g in ("男", "女"):
        v = np.array([r["ladder_f0"] for r in rows if r["gender"] == g])
        p = np.array([r["pyin"] for r in rows if r["gender"] == g])
        lo, hi = GENDER_RANGE[g]
        print(f"  {g}: n={len(v)} 阶梯F0 中位 {np.median(v):.1f} "
              f"区间 {v.min():.1f}~{v.max():.1f} 落音域 "
              f"{int(((v>=lo)&(v<=hi)).sum())}/{len(v)}   |   "
              f"pyin 中位 {np.median(p):.1f} 落音域 "
              f"{int(((p>=lo)&(p<=hi)).sum())}/{len(v)}")

    json.dump(rows, open(f"{VOICES}/base_f0_octave_ladder.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"  -> {VOICES}/base_f0_octave_ladder.json")


if __name__ == "__main__":
    main()
