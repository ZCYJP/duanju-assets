#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""情绪音色音高回正 v10：中转测量法（relay measurement）。

【统一解法：中转测量】
======================
前面 v3~v9 反复失败，根因始终是同一个：
    【f0 测量只在中等音高区可靠】
    - f0 < 100Hz：2 倍频能量压过基频 -> 倍频误锁
      （实证 M02_yan 目标 74.2Hz，三个算法全读 128~152Hz = 2 倍频）
    - f0 > 400Hz：接近 800Hz 搜索上限 -> 读数偏低
      （实证 F02_nu 原生 441Hz，升调 1.6 倍到 706Hz 后读数偏低 24%）
    - 中等区（150~300Hz）：三算法一致，往返误差 0.1~1.1%

所以：不要在极端区测量。把音频先【中转】到 200Hz 附近再测。

    f0_true = f0_mid / K_total
    其中 K_total 是中转用的总变调倍率（解析已知，无需测量）

关键：K_total 是【我们自己设定的解析值】，
      所以中转法的精度只取决于「中转后的测量精度」，
      而中转后必然落在可靠区。

【中转倍率的选取】
=================
1. 先做【粗测】（三算法中位数，快）得到 f0_rough
2. 选 K 使中转后落在 RELAY_F0=200Hz 附近：
       K = RELAY_F0 / f0_rough
   若 K 超出 [0.4, 2.5]，分步执行（每步 ∈ [0.5, 2.0]）
3. 粗测即使错 2 倍（误锁），中转后仍在 100~400Hz，仍在可靠区，
   而 f0_true = f0_mid / K 用的是解析 K，不受粗测误差影响。

【基线与情绪音频同口径】
=======================
base_f0 与情绪 f0 都用中转法测量，
保证「拿同一把尺子量两件东西」。

【回正流程】
===========
1. 中转法测 f0_0（原始情绪音频）与 base（角色基线）
2. r_before = f0_0 / base
3. factor = target_r / r_before，一次变调到位（分步保音质）
4. 中转法测 f0_after，r_after = f0_after / base
5. 小步闭环精修（限幅 0.85~1.18），反馈在可靠区，可信

用法
----
  python3 tmp_emotion_pitch_fix_v10.py --rebuild-base --emotions ""
  python3 tmp_emotion_pitch_fix_v10.py --workers 48 --out-dir .../emotions_fixed
  python3 tmp_emotion_pitch_fix_v10.py --merge
  python3 tmp_emotion_pitch_fix_v10.py --self-check    # 口径自检
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
import librosa

VOICES = "/ssd2/duanju/voices"
EMOTIONS = f"{VOICES}/emotions"
BASE_F0 = f"{VOICES}/base_f0_v10.json"
CAST = "/ssd2/duanju/scripts/cast_all_150.json"

FMIN, FMAX = 50.0, 800.0

# 中转目标音高：位于测量可靠区中部
RELAY_F0 = 200.0
# 中转倍率单步限幅
SUB_LO, SUB_HI = 0.50, 2.0

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

TOL = 0.03
STEP_LO, STEP_HI = 0.85, 1.18
MAX_REFINE = 4
FACTOR_MIN, FACTOR_MAX = 0.15, 3.5


# ------------------------------------------------------------- 估计器
def est_pyin_w(x, sr, frame=2048):
    try:
        f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=FMIN, fmax=FMAX,
                                sr=sr, frame_length=frame,
                                hop_length=max(frame // 4, 64))
    except Exception:
        return 0.0
    v = f0[(f0 > FMIN) & (f0 < FMAX)]
    return float(np.median(v)) if len(v) else 0.0


def _yin_diff(seg, tau_max):
    n = len(seg)
    seg = seg - seg.mean()
    p0 = float(np.dot(seg, seg))
    acf = np.correlate(seg, seg, "full")[n - 1:]
    if len(acf) <= tau_max:
        return None
    sq = np.concatenate(([0.0], np.cumsum(seg ** 2)))
    ptau = sq[n:] - sq[:n]
    d = p0 + ptau[:tau_max + 1] - 2.0 * acf[:tau_max + 1]
    return np.maximum(d, 0.0)


def _parabolic(y, i):
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    if abs(den) < 1e-12:
        return float(i)
    d = 0.5 * (a - c) / den
    return float(i) + d if abs(d) <= 1.0 else float(i)


def est_yin_thr_w(x, sr, frame=4096, threshold=0.15):
    hop = max(frame // 4, 64)
    tau_min = max(int(sr / FMAX), 2)
    tau_max = min(int(sr / FMIN), frame - 1)
    if tau_max <= tau_min + 2:
        return 0.0
    vals = []
    for st in range(0, len(x) - frame, hop):
        seg = x[st:st + frame].astype(np.float64)
        if np.sqrt(np.mean(seg ** 2)) < 0.012:
            continue
        d = _yin_diff(seg, tau_max)
        if d is None:
            continue
        cum = np.cumsum(d[1:])
        taus = np.arange(1, len(d))
        dp = np.ones_like(d)
        dp[1:] = d[1:] * taus / np.maximum(cum, 1e-12)
        below = np.where(dp[tau_min:tau_max + 1] < threshold)[0]
        if len(below) == 0:
            i = int(np.argmin(dp[tau_min:tau_max + 1])) + tau_min
        else:
            i = int(below[0]) + tau_min
            while i + 1 <= tau_max and dp[i + 1] < dp[i]:
                i += 1
        tau = _parabolic(dp, i)
        if tau > 0:
            f = sr / tau
            if FMIN <= f <= FMAX:
                vals.append(f)
    return float(np.median(vals)) if vals else 0.0


def est_ac_w(x, sr, frame=4096):
    hop = max(frame // 4, 64)
    lo, hi = int(sr / FMAX), int(sr / FMIN)
    vals = []
    for st in range(0, len(x) - frame, hop):
        seg = x[st:st + frame].astype(np.float64)
        if np.sqrt(np.mean(seg ** 2)) < 0.012:
            continue
        seg = seg - seg.mean()
        acf = np.correlate(seg, seg, "full")[frame - 1:]
        if hi >= len(acf):
            continue
        s = acf[lo:hi]
        if len(s):
            vals.append(sr / (lo + int(np.argmax(s))))
    return float(np.median(vals)) if vals else 0.0


def measure_raw(x, sr, frame=None):
    """直接测量（三算法中位数）。frame=None 时按音高自适应。"""
    if frame is None:
        frame = 2048
    v = [est_pyin_w(x, sr, frame),
         est_yin_thr_w(x, sr, max(frame, 4096)),
         est_ac_w(x, sr, frame)]
    vv = [a for a in v if a > 0]
    if not vv:
        return 0.0, {"pyin": round(v[0], 1), "yin_thr": round(v[1], 1),
                     "ac": round(v[2], 1)}
    med = float(np.median(vv))
    return med, {"pyin": round(v[0], 1), "yin_thr": round(v[1], 1),
                 "ac": round(v[2], 1),
                 "spread": round(float((max(vv) - min(vv)) / med * 100), 1)}


# ------------------------------------------------------------- 变调
def pitch_shift_once(x, sr, factor):
    """单轮变调，等长输出 + rms 幅度补偿。精度已验证（误差 0.1~1.1%）。"""
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
    """把总 factor 拆成若干温和步骤执行（每步 ∈ [0.50, 2.0]）。"""
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


def measure_relay(x, sr, relay_f0=RELAY_F0):
    """中转测量：把音频中转到 relay_f0 附近再测量，反推真实 f0。

    返回 (f0_true, detail)。
    """
    # 1) 粗测（自适应窗）
    rough, d0 = measure_raw(x, sr, frame=2048)
    if rough <= 0:
        return 0.0, {"rough": 0.0}

    # 2) 中转倍率：让中转后落在 relay_f0
    k = relay_f0 / rough
    k = min(max(k, 0.35), 3.0)
    y, nsteps = pitch_shift_total(x, sr, k)

    # 3) 在中转区测量（此时音高在可靠区，窗长用 2048 足够）
    f0_mid, d1 = measure_raw(y, sr, frame=2048)

    detail = {"rough": round(rough, 1), "relay_k": round(k, 4),
              "relay_steps": nsteps, "f0_mid": round(f0_mid, 1),
              "mid_spread": d1.get("spread")}
    if f0_mid <= 0:
        return 0.0, detail
    f0_true = f0_mid / k
    detail["f0_true"] = round(f0_true, 1)
    # 自洽性：f0_true 应与 rough 同量级
    detail["rel_ratio"] = round(f0_true / rough, 3)
    return f0_true, detail


# ------------------------------------------------------------- 回正
def _match_rms(y, x):
    rms_in = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    rms_out = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if rms_out <= 1e-8 or rms_in <= 1e-8:
        return y
    y = y * (rms_in / rms_out)
    pk = float(np.max(np.abs(y)))
    if pk > 0.99:
        over = pk / 0.99
        y = np.tanh(y * (over / 0.99) / over) * 0.99
        rms_out2 = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
        if rms_out2 > 1e-8:
            y = y * (rms_in / rms_out2)
            pk = float(np.max(np.abs(y)))
            if pk > 0.99:
                y = y * (0.99 / pk)
    return y


def pitch_fix(x, sr, base, target_r):
    info = {"rounds": 0, "stage": "init", "steps": 0}
    f0_before, d0 = measure_relay(x, sr)
    r_before = f0_before / base if base > 0 else 0.0
    info["r_before"] = round(r_before, 3)
    info["f0_before"] = round(f0_before, 1)
    info["m_before"] = d0
    if r_before <= 0:
        info["stage"] = "no_measure"
        return x.astype(np.float32), info

    # 阶段 1：开环一次到位（分步执行）
    factor = min(max(target_r / r_before, FACTOR_MIN), FACTOR_MAX)
    y, ns = pitch_shift_total(x, sr, factor)
    info["steps"] += ns
    info["rounds"] += 1
    info["open_factor"] = round(factor, 4)
    info["stage"] = "openloop"
    f0_now, d1 = measure_relay(y, sr)
    r_now = f0_now / base if base > 0 else 0.0
    info["m_after"] = d1

    # 阶段 2：小步闭环精修（中转测量可靠，反馈可信）
    for _ in range(MAX_REFINE):
        if abs(r_now - target_r) <= TOL:
            break
        need = target_r / r_now if r_now > 0 else 1.0
        step = min(max(need, STEP_LO), STEP_HI)
        if abs(step - 1.0) < 0.01:
            break
        y = pitch_shift_once(y, sr, step)
        info["rounds"] += 1
        f0_now, d1 = measure_relay(y, sr)
        r_now = f0_now / base if base > 0 else 0.0
        info["stage"] = "refine"

    y = _match_rms(y, x)
    f0_after, df = measure_relay(y, sr)
    info["r_after"] = round(f0_after / base, 3) if base > 0 else 0.0
    info["f0_after"] = round(f0_after, 1)
    info["m_final"] = df
    info["in_range"] = bool(abs(r_now - target_r) <= TOL)
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
        f0, d = measure_relay(x, sr)
        out[cid] = round(f0, 1)
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
            "base": base, "target_r": target_r,
            "r_before": info["r_before"], "r_after": info["r_after"],
            "f0_before": info["f0_before"], "f0_after": info["f0_after"],
            "open_factor": info.get("open_factor"),
            "steps": info.get("steps"),
            "rounds": info["rounds"], "stage": info["stage"],
            "in_range": bool(info["in_range"]),
            "dur_before": round(dur_before, 3),
            "dur_after": round(len(y) / sr, 3),
            "m_before": info.get("m_before"),
            "m_after": info.get("m_after"),
            "m_final": info.get("m_final"),
        }
    except Exception as ex:
        return {"path": p, "cid": cid, "emotion": ek,
                "error": f"{type(ex).__name__}: {str(ex)[:200]}"}


def self_check():
    """口径自检：中转法 vs 直接测量 在真实音频上的差异。"""
    print("=" * 84)
    print("中转测量法自检：直接测 vs 中转测（真实音频）")
    print("=" * 84)
    files = sorted(glob.glob(f"{VOICES}/samples/*_voice.wav") +
                   glob.glob(f"{VOICES}/samples_ext/*_voice.wav"))
    rows = []
    for p in files[::10]:
        cid = os.path.basename(p).split("_")[0]
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        fd, _ = measure_raw(x, sr, frame=2048)
        fr, d = measure_relay(x, sr)
        rows.append((cid, fd, fr, d))
    print(f"{'cid':<7}{'直接测':>10}{'中转测':>10}{'比值':>8}"
          f"{'rough':>8}{'relay_k':>9}{'f0_mid':>9}")
    for cid, fd, fr, d in rows:
        print(f"{cid:<7}{fd:10.1f}{fr:10.1f}{(fr/fd if fd else 0):8.3f}"
              f"{d.get('rough',0):8.1f}{d.get('relay_k',0):9.3f}"
              f"{d.get('f0_mid',0):9.1f}")
    a = np.array([r[2] / r[1] for r in rows if r[1] > 0 and r[2] > 0])
    if len(a):
        print(f"\n中转/直接 比值: 中位 {np.median(a):.3f}  "
              f"p90 {np.percentile(np.abs(a-1)*100,90):.1f}%  "
              f"最大偏差 {np.max(np.abs(a-1))*100:.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotions",
                    default="xi,nu,ai,le,ju,jing,yan,you,xiu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--tag", default="v10")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--rebuild-base", action="store_true")
    ap.add_argument("--only-cids", default="")
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--out", default=f"{VOICES}/pitch_fix_report_v10.json")
    args = ap.parse_args()

    if args.self_check:
        self_check()
        return

    if args.merge:
        src = args.out_dir or f"{VOICES}/emotions_fixed"
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
