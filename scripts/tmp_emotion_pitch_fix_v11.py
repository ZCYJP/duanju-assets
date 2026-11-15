"""v11：基线锚定的音高回正（重构版）。

背景（v3 -> v10 七代演进后的根本发现）
----------------------------------------
所有时域 f0 估计器（pyin / yin_thr / ac）在超低音男声上会锁到 2f/3f/4f 高次谐波。
实证：M02_yan 真实 f0 ≈ 90Hz，pyin 读 920.9Hz（10.1 倍）。
后果：v7~v10 依据这些错误读数，把本已正常的音频强行降调 3~10 倍，
      造成音质灾难；v8 放宽 FACTOR_MIN 到 0.15 更是批量损坏超低音男声。

根本解法：基线锚定（base anchoring）
------------------------------------
结构事实：情绪音频与角色基线出自同一 TTS 音色设计，音高必然同量级。
用户给的 9 情绪音高倍率表也限定在基线的 0.82~1.40 倍。
=> 把 f0 搜索限制在 [base*0.55, base*2.30] 的窄带，
   倍频歧义被搜索带本身消除（带宽 < 2 个八度）。

已验证精度（真实人声 round-trip，280 次）：
   锚定法  中位 1.79%  p90 8.26%  最大 21.3%
   pyin    中位 5.16%  p90 17.10%  最大 439%

流程
----
1. 基线：每角色测一次（锚定带以 200Hz 为中心的宽带自举）
2. 情绪：锚定法测 r_before = f0_emo / base
3. 目标：r_target = 情绪倍率表中值
4. 变调：factor = r_target / r_before（解析式一次到位，分步执行保音质）
5. 校验：锚定法复测 r_after，|r_after - r_target| <= TOL 判定达标
6. 若未达标：小步闭环精修（反馈在锚定带内，可信）

用法
----
  python3 tmp_emotion_pitch_fix_v11.py --rebuild-base
  python3 tmp_emotion_pitch_fix_v11.py --workers 48 --out-dir .../emotions_fixed_v11
  python3 tmp_emotion_pitch_fix_v11.py --merge --tag v11
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import soundfile as sf

VOICES = "/ssd2/duanju/voices"
EMOTIONS = f"{VOICES}/emotions"
BASE_F0 = f"{VOICES}/base_f0_v11.json"

# 锚定带（相对基线）
BAND_LO, BAND_HI = 0.55, 2.30
# 基线自举的宽带（相对 200Hz 中心）
BASE_LO, BASE_HI = 0.35, 3.20

# 9 情绪音高倍率区间（用户给定的声音特征表）
TARGET_RATIO = {
    "xi":   (1.00, 1.30),
    "nu":   (0.95, 1.30),
    "ai":   (0.82, 1.02),
    "le":   (0.95, 1.12),
    "ju":   (1.00, 1.30),
    "jing": (1.05, 1.40),
    "yan":  (0.85, 1.05),
    "you":  (0.90, 1.08),
    "xiu":  (0.98, 1.22),
}

TOL = 0.04
STEP_LO, STEP_HI = 0.85, 1.18
MAX_REFINE = 4
FACTOR_MIN, FACTOR_MAX = 0.30, 3.0
SUB_LO, SUB_HI = 0.50, 2.0


# ------------------------------------------------------------- 谱
def frame_spectrum(x, sr, n_fft=4096, hop=1024):
    if len(x) < n_fft:
        x = np.pad(x, (0, n_fft - len(x)))
    fr = librosa.util.frame(x, frame_length=n_fft, hop_length=hop)
    w = np.hanning(n_fft).astype(np.float32)
    S = np.abs(np.fft.rfft(fr * w[:, None], axis=0))
    return S.T, np.fft.rfftfreq(n_fft, 1.0 / sr)


def whole_spectrum(x, sr):
    n_fft = 1 << int(np.ceil(np.log2(max(len(x), 4096))))
    w = np.hanning(len(x)).astype(np.float32)
    X = np.abs(np.fft.rfft(x * w, n=n_fft))
    return X, np.fft.rfftfreq(n_fft, 1.0 / sr)


# ------------------------------------------------------------- 梳打分
def comb_score_grid(S, freqs, grid, comb_max=2200.0, weight="inv_sqrt",
                    norm_pow=0.5):
    if S.ndim == 1:
        S = S[None, :]
        one = True
    else:
        one = False
    H = np.maximum((comb_max / grid).astype(int), 1)
    hmax = int(H.max())
    if S.shape[0] > 40:
        idx = np.linspace(0, S.shape[0] - 1, 40).astype(int)
        S = S[idx]
    total = np.zeros((S.shape[0], len(grid)))
    for h in range(1, hmax + 1):
        mask = H >= h
        if not np.any(mask):
            continue
        f_h = grid[mask] * h
        pos = np.interp(f_h, freqs, np.arange(len(freqs)))
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, len(freqs) - 1)
        frac = pos - lo
        amp = S[:, lo] * (1 - frac) + S[:, hi] * frac
        total[:, mask] += (1.0 / np.sqrt(h) if weight == "inv_sqrt"
                           else 1.0 / h) * amp
    denom = np.maximum(H.astype(np.float64) ** norm_pow, 1e-9)
    score = total / denom[None, :]
    return score[0] if one else score


def refine_peak(grid, score):
    i = int(np.argmax(score))
    if i <= 0 or i >= len(grid) - 1:
        return float(grid[i])
    y0, y1, y2 = score[i - 1], score[i], score[i + 1]
    den = y0 - 2 * y1 + y2
    d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
    if abs(d) > 1.0:
        d = 0.0
    return float(grid[i] + d * (grid[1] - grid[0]))


# ------------------------------------------------------------- f0 估计
def _comb_one(x, sr, grid, weight, norm_pow):
    S, freqs = frame_spectrum(x, sr)
    sc = comb_score_grid(S, freqs, grid, weight=weight, norm_pow=norm_pow)
    idx = np.argmax(sc, axis=1)
    f_med = float(np.median(grid[idx]))
    df = grid[1] - grid[0]
    g2 = np.arange(max(grid[0], f_med * 0.88),
                   min(grid[-1], f_med * 1.12) + df * 0.5, df * 0.25)
    if len(g2) >= 5:
        Sw, fw = whole_spectrum(x, sr)
        sc2 = comb_score_grid(Sw, fw, g2, weight=weight, norm_pow=norm_pow)
        return refine_peak(g2, sc2)
    return f_med


def _pyin_folded(x, sr, lo, hi):
    try:
        fp, _, _ = librosa.pyin(x.astype(np.float32), fmin=40, fmax=1200, sr=sr,
                                frame_length=2048, hop_length=512)
        v = fp[(fp > 40) & (fp < 1200)]
        if not len(v):
            return 0.0
        p = float(np.median(v))
        k = 0
        while p / (2 ** (k + 1)) >= lo and k < 6:
            k += 1
        while p / (2 ** k) > hi and k < 6:
            k += 1
        return p / (2 ** k)
    except Exception:
        return 0.0


CFGS = (("inv_sqrt", 0.5), ("inv", 0.0), ("inv_sqrt", 0.3))


def f0_anchored(x, sr, base, band=(BAND_LO, BAND_HI), df=0.5):
    """基线锚定 f0。返回 (f0, detail)。"""
    lo = max(40.0, base * band[0])
    hi = min(1400.0, base * band[1])
    if hi <= lo:
        return 0.0, {}
    grid = np.arange(lo, hi + df, df)
    vals = []
    for w, np_ in CFGS:
        f = _comb_one(x, sr, grid, w, np_)
        if f > 0:
            vals.append(f)
    p = _pyin_folded(x, sr, lo, hi)
    if p > 0:
        vals.append(p)
    if not vals:
        return 0.0, {}
    return float(np.median(vals)), {"vals": [round(v, 1) for v in vals]}


def f0_base(x, sr):
    """基线自举：宽带 + 多配置中位数。"""
    grid = np.arange(max(40.0, 200.0 * BASE_LO),
                     min(1200.0, 200.0 * BASE_HI) + 0.5, 0.5)
    vals = []
    for w, np_ in CFGS:
        f = _comb_one(x, sr, grid, w, np_)
        if f > 0:
            vals.append(f)
    try:
        fp, _, _ = librosa.pyin(x.astype(np.float32), fmin=45, fmax=800, sr=sr,
                                frame_length=2048, hop_length=512)
        v = fp[(fp > 45) & (fp < 800)]
        if len(v):
            vals.append(float(np.median(v)))
    except Exception:
        pass
    if not vals:
        return 0.0
    return float(np.median(vals))


# ------------------------------------------------------------- 变调
def pitch_shift_once(x, sr, factor):
    """单轮变调，等长输出 + rms 幅度补偿。精度已验证（0.1~1.1%）。"""
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


def pitch_shift_total(x, sr, factor):
    """总 factor 拆成若干温和步骤（每步 ∈ [0.50, 2.0]）。"""
    if abs(factor - 1.0) < 0.005:
        return x.astype(np.float32), 0
    steps = []
    f = factor
    guard = 0
    while f < SUB_LO and guard < 12:
        steps.append(SUB_LO)
        f /= SUB_LO
        guard += 1
    while f > SUB_HI and guard < 12:
        steps.append(SUB_HI)
        f /= SUB_HI
        guard += 1
    steps.append(f)
    y = x.astype(np.float32)
    for s in steps:
        y = pitch_shift_once(y, sr, s)
    return y, len(steps)


def _match_rms(y, x):
    rms_in = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    rms_out = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if rms_out <= 1e-8 or rms_in <= 1e-8:
        return y
    y = y * (rms_in / rms_out)
    pk = float(np.max(np.abs(y)))
    if pk > 0.99:
        y = np.tanh(y * 1.02) * 0.99
    return y


# ------------------------------------------------------------- 回正
def pitch_fix(x, sr, base, target_r):
    info = {"rounds": 0}
    f0_b, d0 = f0_anchored(x, sr, base)
    r_b = f0_b / base if base > 0 else 0.0
    info["r_before"] = round(r_b, 3)
    info["f0_before"] = round(f0_b, 1)
    info["m_before"] = d0
    if r_b <= 0:
        info["stage"] = "no_measure"
        return x.astype(np.float32), info

    factor = min(max(target_r / r_b, FACTOR_MIN), FACTOR_MAX)
    y, ns = pitch_shift_total(x, sr, factor)
    info["steps"] = ns
    info["rounds"] = 1
    info["open_factor"] = round(factor, 4)
    info["stage"] = "openloop"
    f0_n, d1 = f0_anchored(y, sr, base)
    r_n = f0_n / base if base > 0 else 0.0
    info["m_after"] = d1

    for _ in range(MAX_REFINE):
        if abs(r_n - target_r) <= TOL:
            break
        need = target_r / r_n if r_n > 0 else 1.0
        step = min(max(need, STEP_LO), STEP_HI)
        if abs(step - 1.0) < 0.01:
            break
        y = pitch_shift_once(y, sr, step)
        info["rounds"] += 1
        f0_n, d1 = f0_anchored(y, sr, base)
        r_n = f0_n / base if base > 0 else 0.0
        info["stage"] = "refine"

    y = _match_rms(y, x)
    f0_a, df = f0_anchored(y, sr, base)
    info["r_after"] = round(f0_a / base, 3) if base > 0 else 0.0
    info["f0_after"] = round(f0_a, 1)
    info["m_final"] = df
    info["in_range"] = bool(abs(r_n - target_r) <= TOL)
    return y, info


# ------------------------------------------------------------- 基线
def rebuild_base_f0():
    files = sorted(glob.glob(f"{VOICES}/samples/*_voice.wav") +
                   glob.glob(f"{VOICES}/samples_ext/*_voice.wav"))
    out = {}
    t0 = time.time()
    for i, p in enumerate(files):
        cid = os.path.basename(p).split("_")[0]
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        out[cid] = round(f0_base(x, sr), 1)
        if (i + 1) % 25 == 0:
            print(f"  base {i+1}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(BASE_F0, "w", encoding="utf-8"), indent=2)
    print(f"[base] {len(out)} 条 -> {BASE_F0}", flush=True)
    return out


# ------------------------------------------------------------- 并行
_G = {}


def _init_worker(out_dir, base_path):
    _G["out_dir"] = out_dir
    _G["base"] = json.load(open(base_path, encoding="utf-8"))


def _work(job):
    ek, p, target_r, lo_r, hi_r, dry = job
    cid = os.path.basename(p).split("_")[0]
    base = _G["base"].get(cid, 0.0)
    if base <= 0:
        return None
    try:
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        dur_before = len(x) / sr
        y, info = pitch_fix(x, sr, base, target_r)
        dst = os.path.join(_G["out_dir"], os.path.relpath(p, EMOTIONS))
        if not dry:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            sf.write(dst, y, sr)
        return {
            "path": p, "dst": dst, "cid": cid, "emotion": ek,
            "base": base, "target_r": target_r, "lo_r": lo_r, "hi_r": hi_r,
            "r_before": info["r_before"], "r_after": info["r_after"],
            "f0_before": info["f0_before"], "f0_after": info["f0_after"],
            "open_factor": info.get("open_factor"),
            "steps": info.get("steps"), "rounds": info["rounds"],
            "stage": info["stage"], "in_range": bool(info["in_range"]),
            "dur_before": round(dur_before, 3),
            "dur_after": round(len(y) / sr, 3),
            "m_before": info.get("m_before"), "m_after": info.get("m_after"),
            "m_final": info.get("m_final"),
        }
    except Exception as ex:
        return {"path": p, "cid": cid, "emotion": ek,
                "error": f"{type(ex).__name__}: {str(ex)[:200]}"}


# ------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotions",
                    default="xi,nu,ai,le,ju,jing,yan,you,xiu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--tag", default="v11")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--rebuild-base", action="store_true")
    ap.add_argument("--only-cids", default="")
    ap.add_argument("--out", default=f"{VOICES}/pitch_fix_report_v11.json")
    args = ap.parse_args()

    if args.merge:
        src = args.out_dir or f"{VOICES}/emotions_fixed_v11"
        if not args.no_backup:
            bak = f"{VOICES}/emotions_backup_{args.tag}"
            if os.path.exists(bak):
                print(f"[merge] 备份已存在，跳过：{bak}")
            else:
                os.makedirs(bak, exist_ok=True)
                subprocess.run(["cp", "-a", f"{EMOTIONS}/.", f"{bak}/"],
                               check=True)
                print(f"[merge] 已备份原目录 -> {bak}")
        n = 0
        for p in glob.glob(f"{src}/*/*.wav"):
            dst = os.path.join(EMOTIONS, os.path.relpath(p, src))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(p, dst)
            n += 1
        print(f"[merge] {n} 个文件已合并回 {EMOTIONS}")
        return

    if args.rebuild_base:
        rebuild_base_f0()

    base_map = json.load(open(BASE_F0, encoding="utf-8"))
    emos = [e.strip() for e in args.emotions.split(",") if e.strip()]
    out_dir = args.out_dir or EMOTIONS
    only = set(c.strip() for c in args.only_cids.split(",") if c.strip())

    jobs = []
    for ek in emos:
        lo_r, hi_r = TARGET_RATIO.get(ek, (0.9, 1.3))
        target_r = (lo_r + hi_r) / 2.0
        files = sorted(glob.glob(f"{EMOTIONS}/*/*_{ek}.wav"))
        if args.limit:
            files = files[:args.limit]
        for p in files:
            cid = os.path.basename(p).split("_")[0]
            if only and cid not in only:
                continue
            jobs.append((ek, p, target_r, lo_r, hi_r, args.dry_run))

    print(f"[plan] {len(jobs)} 任务 / {args.workers} 进程", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(out_dir, BASE_F0)) as ex:
        for i, r in enumerate(ex.map(_work, jobs, chunksize=1)):
            if r:
                results.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    summary = {}
    for ek in emos:
        rr = [r for r in results if r.get("emotion") == ek and "r_after" in r]
        if not rr:
            continue
        rb = np.array([r["r_before"] for r in rr])
        ok = sum(1 for r in rr if r["in_range"])
        summary[ek] = {
            "n": len(rr), "in_range": ok,
            "rate": round(ok / len(rr) * 100, 1),
            "r_before_mean": round(float(rb.mean()), 3),
            "r_after_mean": round(float(np.mean([r["r_after"] for r in rr])), 3),
            "r_after_median": round(float(np.median([r["r_after"] for r in rr])), 3),
        }
        print(f"[done] {ek}: n={len(rr)} 达标 {ok}/{len(rr)} "
              f"({ok/len(rr)*100:.1f}%)  倍率均值 {rb.mean():.2f} -> "
              f"{np.mean([r['r_after'] for r in rr]):.2f}", flush=True)

    json.dump({"summary": summary, "items": results},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False,
              indent=2)
    print(f"-> {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
