#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""情绪音色音高回正 v8：修正 factor 下限 clamp + 分步降调 + 低音友好测量。

【v7 未达标 29 条的真正根因（终于定位）】
=========================================
不是测量不准，是【我自己的 factor 下限 clamp 造成的】。

v7 代码：factor = min(max(target_r / r_before, 0.28), 3.5)
                                        ^^^^ 这个 0.28 是硬下限
实测：
    M13 nu  base=73.7  r_before=4.523  target=1.125
         需要 factor = 1.125/4.523 = 0.249  < 0.28  <- 被 clamp！
         实际只降到 f0 = 333 * 0.28 = 93.3Hz
         而目标 = 73.7 * 1.125 = 82.9Hz
         -> 降调不足，残留 r=1.266（实测 1.442，含测量误差）
    M02 yan base=80.8  r_before=5.121  target=0.950
         需要 factor = 0.950/5.121 = 0.186  << 0.28  <- 差得更远

漂移最严重的角色反而被 clamp 卡住，这是个典型的自伤设计：
clamp 本意是防止过度降调毁音质，却在最需要降调的场景挡住了。

【v8 的三处修正】
===============
1. factor 下限 0.28 -> 0.15，且【分步执行】保音质：
       总 factor 拆成若干步，每步 ∈ [0.50, 2.0]
       例：0.186 -> 0.50 × 0.50 × 0.744
       避免单次 factor=0.186 的 resample 到 129kHz 的极端操作。
2. FMIN 65 -> 50 Hz：
       超低音角色 base 67~75Hz，哀/厌目标 f0 = 0.92*67 = 61.6Hz，
       原 FMIN=65 会把真实基频直接排除在搜索窗外。
       实测新配置对正常音域角色读数无影响
       （tmp_lowf0_search.py：F01/M01/M07/F30 全部一致）。
3. 低音自适应窗长：
       期望 f0 < 110Hz 时用 8192（341ms，73Hz 有 25 个周期），
       否则 2048/1024。

【保持不变】
==========
- 三算法中位数交叉（pyin / yin_thr / ac），无 octave_fix
- 不使用升调校准（会把测量推向失真区）
- 开环一次到位 + 小步闭环精修
- 基线同口径重算

用法
----
  python3 tmp_emotion_pitch_fix_v8.py --rebuild-base --emotions "" 
  python3 tmp_emotion_pitch_fix_v8.py --workers 48 --out-dir .../emotions_fixed
  python3 tmp_emotion_pitch_fix_v8.py --merge
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
BASE_F0 = f"{VOICES}/base_f0_v8.json"
CAST = "/ssd2/duanju/scripts/cast_all_150.json"

FMIN, FMAX = 50.0, 800.0

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
# 总 factor 的允许范围（修正：下限 0.28 -> 0.15）
FACTOR_MIN, FACTOR_MAX = 0.15, 3.5
# 单步变调限幅（分步执行，每步温和）
SUB_LO, SUB_HI = 0.50, 2.0


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


def measure_f0(x, sr, expect_f0=None):
    """三算法中位数交叉测量，窗长按预期 f0 自适应。"""
    if expect_f0 is None:
        frame = 2048
    elif expect_f0 < 110:
        frame = 8192
    elif expect_f0 < 200:
        frame = 4096
    else:
        frame = 2048
    v = [est_pyin_w(x, sr, frame),
         est_yin_thr_w(x, sr, max(frame, 4096)),
         est_ac_w(x, sr, frame)]
    detail = {"pyin": round(v[0], 1), "yin_thr": round(v[1], 1),
              "ac": round(v[2], 1), "frame": frame}
    vv = [a for a in v if a > 0]
    if not vv:
        return 0.0, detail
    detail["median"] = round(float(np.median(vv)), 1)
    detail["spread"] = round(float((max(vv) - min(vv)) /
                                   np.median(vv) * 100), 1)
    return float(np.median(vv)), detail


# ------------------------------------------------------------- 变调
def pitch_shift_once(x, sr, factor):
    """单轮变调（factor<1 降调、>1 升调），等长输出 + rms 幅度补偿。

    精度已验证（tmp_shift_compose.py）：降调误差 0.1~1.1%。
    """
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
    """把总 factor 拆成若干温和步骤执行（保音质）。

    例：factor=0.186 -> 0.50 x 0.50 x 0.744
    每步 factor ∈ [0.50, 2.0]，避免单次极端 resample。
    """
    if abs(factor - 1.0) < 0.005:
        return x.astype(np.float32), 0
    steps = []
    f = factor
    guard = 0
    while f < SUB_LO and guard < 10:
        steps.append(SUB_LO)
        f /= SUB_LO
        guard += 1
    while f > SUB_HI and guard < 10:
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
    f0_before, d0 = measure_f0(x, sr)
    r_before = f0_before / base if base > 0 else 0.0
    info["r_before"] = round(r_before, 3)
    info["f0_before"] = round(f0_before, 1)
    info["m_before"] = d0
    if r_before <= 0:
        info["stage"] = "no_measure"
        return x.astype(np.float32), info

    # 阶段 1：开环一次到位（分步执行）
    factor = min(max(target_r / r_before, FACTOR_MIN), FACTOR_MAX)
    y, nsteps = pitch_shift_total(x, sr, factor)
    info["rounds"] = 1
    info["steps"] = nsteps
    info["open_factor"] = round(factor, 4)
    info["stage"] = "openloop"
    f0_now, d1 = measure_f0(y, sr, expect_f0=base * target_r)
    r_now = f0_now / base if base > 0 else 0.0
    info["m_after"] = d1

    # 阶段 2：小步闭环精修
    for _ in range(MAX_REFINE):
        if abs(r_now - target_r) <= TOL:
            break
        need = target_r / r_now if r_now > 0 else 1.0
        step = min(max(need, STEP_LO), STEP_HI)
        if abs(step - 1.0) < 0.01:
            break
        y = pitch_shift_once(y, sr, step)
        info["rounds"] += 1
        f0_now, d1 = measure_f0(y, sr, expect_f0=base * target_r)
        r_now = f0_now / base if base > 0 else 0.0
        info["stage"] = "refine"

    # 阶段 3：残留过大 -> 用更准的初测重新开环
    if abs(r_now - target_r) > 0.08:
        f0b2, _ = measure_f0(x, sr)
        if f0b2 > 0:
            fac2 = min(max(target_r / (f0b2 / base), FACTOR_MIN), FACTOR_MAX)
            y2, ns2 = pitch_shift_total(x, sr, fac2)
            f02, d2 = measure_f0(y2, sr, expect_f0=base * target_r)
            r2 = f02 / base if base > 0 else 0.0
            if abs(r2 - target_r) < abs(r_now - target_r):
                y, r_now, d1 = y2, r2, d2
                info["stage"] = "reopen"
                info["rounds"] += 1
                info["steps"] += ns2

    y = _match_rms(y, x)
    f0_after, df = measure_f0(y, sr, expect_f0=base * target_r)
    info["r_after"] = round(f0_after / base, 3) if base > 0 else 0.0
    info["f0_after"] = round(f0_after, 1)
    info["m_final"] = df
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
        f0, d = measure_f0(x, sr)
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
        r_after = info["r_after"]
        ok = bool(lo_r - TOL <= r_after <= hi_r + TOL)
        dst = os.path.join(_G["out_dir"], os.path.relpath(p, EMOTIONS))
        if not dry:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            sf.write(dst, y, sr)
        return {
            "path": p, "dst": dst, "cid": cid, "emotion": ek,
            "base": base, "target_r": target_r,
            "r_before": info["r_before"], "r_after": r_after,
            "f0_before": info["f0_before"], "f0_after": info["f0_after"],
            "open_factor": info.get("open_factor"),
            "steps": info.get("steps"),
            "rounds": info["rounds"], "stage": info["stage"],
            "in_range": ok,
            "dur_before": round(dur_before, 3),
            "dur_after": round(len(y) / sr, 3),
            "m_before": info.get("m_before"),
            "m_after": info.get("m_after"),
            "m_final": info.get("m_final"),
        }
    except Exception as ex:
        return {"path": p, "cid": cid, "emotion": ek,
                "error": f"{type(ex).__name__}: {str(ex)[:200]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotions",
                    default="xi,nu,ai,le,ju,jing,yan,you,xiu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--tag", default="v8")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--rebuild-base", action="store_true")
    ap.add_argument("--only-cids", default="",
                    help="只处理指定 cid（逗号分隔），用于定点修复")
    ap.add_argument("--out", default=f"{VOICES}/pitch_fix_report_v8.json")
    args = ap.parse_args()

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
        ra = np.array([r["r_after"] for r in rr])
        ok = sum(1 for r in rr if r["in_range"])
        summary[ek] = {
            "n": len(rr), "in_range": ok,
            "rate": round(ok / len(rr) * 100, 1),
            "r_before_mean": round(float(rb.mean()), 3),
            "r_after_mean": round(float(ra.mean()), 3),
            "r_after_median": round(float(np.median(ra)), 3),
            "rounds_mean": round(float(np.mean([r["rounds"] for r in rr])), 1),
            "steps_mean": round(float(np.mean([r.get("steps", 0) for r in rr])), 1),
        }
        print(f"[done] {ek}: n={len(rr)} 达标 {ok}/{len(rr)} "
              f"({ok/len(rr)*100:.1f}%)  倍率 {rb.mean():.2f}->{ra.mean():.2f}  "
              f"中位 {np.median(ra):.3f}", flush=True)

    json.dump({"summary": summary, "items": results},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False,
              indent=2)
    print(f"-> {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
