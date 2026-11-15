#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""情绪音色音高回正 v7（最终版）：多算法中位数交叉 + 开环一次到位。

【根因总结（三个版本的实测教训）】
=================================
v3/v4（自相关 + octave_fix）：闭环发散，极端案例 r 升到 4.5~5.0
    原因：octave_fix 倍频折叠在谐波信号上 p90 误差 35.19%、最大 50%
          （tmp_f0_robust.py 六算法评估）；且自相关会锁到高次谐波。

v5（pYIN + 升调校准）：F17 失败，r 修到 0.735（预期 1.125）
    原因：【升调校准把测量推向失真区】。
          tmp_shift_compose.py 实测 F02_nu 原生 f0 已 441Hz，
          升调 1.6 倍后到 706Hz 撞 700Hz 搜索上限，读数偏低 24%。
          也就是说「升调方向更可靠」这个结论只在【基线音高附近】成立，
          对已经严重漂移到 300~450Hz 的情绪音频不成立。

v7 的口径
=========
1. 【禁止 octave_fix】—— 六算法评估已证其有害。
2. 【禁止升调校准】—— 会撞测量上限。
3. 【三算法中位数交叉】pyin / yin_thr(自研绝对阈值 YIN) / ac 三者取中位数：
       三者在真实语音上的一致性远好于任一单算法，
       且中位数天然抗单算法的偶发误锁。
       lab 实测（16 条真实语音 × 6 个倍率）：
         ac      中位 7.3%  p90 189.0%  最大 886.2%
         pyin    中位 1.8%  p90  26.1%  最大 1108.6%
         yin_thr 中位 3.4%  p90  62.4%  最大 190.5%
4. 【基线同口径】base 也用三算法中位数重算。
5. 【开环一次到位 + 小步闭环精修】
   变调算子已验证精确（tmp_shift_compose.py B 组：降调误差 0.1~1.1%；
   C 组复合变调误差 0.2~5%），所以开环一次到位是可靠的，
   闭环只做小步修正（限幅 0.85~1.18）以吸收测量残差。
6. 【变调后重新升回原高做交叉校验】（可选第二意见）

用法
----
  python3 tmp_emotion_pitch_fix_v7.py --rebuild-base --emotions nu --limit 30 --dry-run
  python3 tmp_emotion_pitch_fix_v7.py --workers 32 --out-dir /ssd2/duanju/voices/emotions_fixed
  python3 tmp_emotion_pitch_fix_v7.py --merge
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
BASE_F0 = f"{VOICES}/base_f0_v7.json"
CAST = "/ssd2/duanju/scripts/cast_all_150.json"

FMIN, FMAX = 65.0, 800.0

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


# ------------------------------------------------------------- 估计器
def est_ac(x, sr, fmin=FMIN, fmax=FMAX):
    """自相关（无倍频修正）。"""
    frame = int(sr * 0.04)
    hop = int(sr * 0.02)
    lo, hi = int(sr / fmax), int(sr / fmin)
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
        if len(s) == 0:
            continue
        vals.append(sr / (lo + int(np.argmax(s))))
    return float(np.median(vals)) if vals else 0.0


def est_pyin(x, sr, fmin=FMIN, fmax=FMAX):
    try:
        f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=fmin, fmax=fmax,
                                sr=sr, frame_length=1024, hop_length=256)
    except Exception:
        return 0.0
    v = f0[(f0 > fmin) & (f0 < fmax)]
    return float(np.median(v)) if len(v) else 0.0


def _yin_diff(seg, tau_max):
    """YIN 差分函数 d(tau)（用自相关加速）。"""
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
    delta = 0.5 * (a - c) / den
    if abs(delta) > 1.0:
        return float(i)
    return float(i) + delta


def est_yin_thr(x, sr, fmin=FMIN, fmax=FMAX, threshold=0.15):
    """自研 YIN：绝对阈值 + 首个谷（YIN 论文原设计，专治八度错误）。"""
    frame, hop = 1024, 256
    tau_min = max(int(sr / fmax), 2)
    tau_max = min(int(sr / fmin), frame - 1)
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
        dprime = np.ones_like(d)
        dprime[1:] = d[1:] * taus / np.maximum(cum, 1e-12)
        below = np.where(dprime[tau_min:tau_max + 1] < threshold)[0]
        if len(below) == 0:
            i = int(np.argmin(dprime[tau_min:tau_max + 1])) + tau_min
        else:
            i = int(below[0]) + tau_min
            while i + 1 <= tau_max and dprime[i + 1] < dprime[i]:
                i += 1
        tau = _parabolic(dprime, i)
        if tau <= 0:
            continue
        f = sr / tau
        if fmin <= f <= fmax:
            vals.append(f)
    return float(np.median(vals)) if vals else 0.0


ESTIMATORS = (("pyin", est_pyin), ("yin_thr", est_yin_thr), ("ac", est_ac))


def measure_f0(x, sr):
    """三算法中位数交叉测量。返回 (f0, detail)。"""
    detail = {}
    vals = []
    for name, fn in ESTIMATORS:
        v = fn(x, sr)
        detail[name] = round(v, 1)
        if v > 0:
            vals.append(v)
    if not vals:
        return 0.0, detail
    arr = np.array(vals)
    detail["median"] = round(float(np.median(arr)), 1)
    detail["spread"] = round(float((arr.max() - arr.min()) /
                                   np.median(arr) * 100), 1)
    return float(np.median(arr)), detail


# ------------------------------------------------------------- 变调
def pitch_shift_once(x, sr, factor):
    """单轮变调（factor<1 降调、>1 升调），等长输出 + rms 幅度补偿。

    精度已验证（tmp_shift_compose.py）：
      降调 0.85/0.70/0.60/0.50 后实测比值 0.853/0.701/0.607/0.501（误差 0.1~1.1%）
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


def _match_rms(y, x):
    """最终 rms 对齐 + tanh 软限幅。"""
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


def pitch_fix(x, sr, base, target_r, lo_r, hi_r):
    info = {"rounds": 0, "stage": "init"}
    f0_before, d0 = measure_f0(x, sr)
    r_before = f0_before / base if base > 0 else 0.0
    info["r_before"] = round(r_before, 3)
    info["f0_before"] = round(f0_before, 1)
    info["m_before"] = d0
    if r_before <= 0:
        info["stage"] = "no_measure"
        return x.astype(np.float32), info

    # 阶段 1：开环一次到位（算子精确，一次到位最干净）
    factor = min(max(target_r / r_before, 0.28), 3.5)
    y = pitch_shift_once(x, sr, factor)
    info["rounds"] = 1
    info["open_factor"] = round(factor, 4)
    info["stage"] = "openloop"
    f0_now, d1 = measure_f0(y, sr)
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
        f0_now, d1 = measure_f0(y, sr)
        r_now = f0_now / base if base > 0 else 0.0
        info["stage"] = "refine"

    # 阶段 3：残留过大 -> 重新开环（用更准的初测值）
    if abs(r_now - target_r) > 0.08:
        y2 = pitch_shift_once(x, sr, factor)
        f02, d2 = measure_f0(y2, sr)
        r2 = f02 / base if base > 0 else 0.0
        if abs(r2 - target_r) < abs(r_now - target_r):
            y, r_now, d1 = y2, r2, d2
            info["stage"] = "reopen"
            info["rounds"] += 1

    y = _match_rms(y, x)
    f0_after, df = measure_f0(y, sr)
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
        y, info = pitch_fix(x, sr, base, target_r, lo_r, hi_r)
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
    ap.add_argument("--tag", default="v7")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--rebuild-base", action="store_true")
    ap.add_argument("--out", default=f"{VOICES}/pitch_fix_report_v7.json")
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

    jobs = []
    for ek in emos:
        lo_r, hi_r = TARGET_RATIO.get(ek, (0.9, 1.3))
        target_r = (lo_r + hi_r) / 2.0
        files = sorted(glob.glob(f"{EMOTIONS}/*/*_{ek}.wav"))
        if args.limit:
            files = files[:args.limit]
        for p in files:
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
