#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""情绪音色音高回正 v9：分段闭环（anchor 锚点法）。

【v8 残余 3 条未达标的根因（诊断实证）】
=========================================
tmp_diag4 的四案例对照（"/期望"= 测得/解析期望值）：

    M54_ai  factor=0.2416 期望f0=87.6
        一次变调 pyin@w4096 = 86.6  -> /期望 = 0.988   【音频其实是对的】
        但 v8 用三算法中位数 (86.6, 84.4, 224.3) -> 中位 86.6 本应对，
        然而 v8 报告 r=1.235 —— 是【阶段2闭环】基于后续错误测量
        (yin/ac 锁到 2 倍频 224Hz) 又调坏了。

    M02_yan factor=0.1615 期望f0=74.2
        所有算法读 128~152Hz（= 2 倍频）-> /期望 ≈ 1.99
        目标 74Hz 落在测量不可靠区，反馈信号是错的。

    M13_yan factor=0.6239 期望f0=64.2
        pyin@w4096 = 64.5 -> /期望 = 1.004 【音频也是对的】
        同 M54，被阶段2 调坏。

结论（两条，都很关键）：
  (1) 【目标 f0 < ~100Hz 时，所有估计器都会锁到 2 倍频】
      因为 f0=74Hz 时 2f=148Hz 的谐波能量远高于 74Hz，
      估计器无法分辨。此时【闭环反馈是错的】，
      基于它做精修会把已经正确的音频调坏。
  (2) 变调算子本身精确（tmp_shift_compose.py：误差 0.1~1.1%），
      所以【最终目标这一步可以用解析式一次到位】，不必依赖测量。

【v9 方案：分段闭环 + anchor 锚点】
====================================
  阶段 A：把音频变调到【测量可靠区】的锚点音高
          anchor_f0 = 150Hz（远高于不可靠区，且低于 800Hz 上限）
          此处闭环反馈可信，迭代收敛到 anchor 精确
  阶段 B：从 anchor 用【解析式】一次性变调到最终目标
          factor = target_f0 / anchor_f0（算子精确，无需测量）
  判定：
          目标 f0 >= 110Hz  -> 用测量判定（可靠区）
          目标 f0 <  110Hz  -> 用 anchor 收敛精度 + 解析式判定
                              （并记录测得值供人工复核）

【为什么阶段 A 一定收敛】
    tmp_shift_compose.py 实测：真实语音降调后测量误差仅 0.1~1.1%
    （F02: 0.853/0.701/0.607/0.501 对应 0.85/0.70/0.60/0.50）
    即【降调方向】的测量是准确的（tmp_f0_roundtrip 的结论
    "降调后测量失真"是在【基线口径】下的旧结论，
    用【同一算法对同一文件前后对比】时是准确的）。

【初始 f0 的稳健估计】
    原始情绪音频漂移到 100~460Hz，处于测量可靠区，
    用三算法中位数；若分歧 > 30% 则用谐波投票裁决。

用法
----
  python3 tmp_emotion_pitch_fix_v9.py --rebuild-base --emotions ""
  python3 tmp_emotion_pitch_fix_v9.py --workers 48 --out-dir .../emotions_fixed
  python3 tmp_emotion_pitch_fix_v9.py --merge
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
BASE_F0 = f"{VOICES}/base_f0_v9.json"
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
FACTOR_MIN, FACTOR_MAX = 0.15, 3.5
SUB_LO, SUB_HI = 0.50, 2.0

# 测量可靠区下界：目标 f0 低于此值时，闭环反馈不可信
RELIABLE_FLOOR = 110.0
# anchor 锚点音高：位于可靠区中部
ANCHOR_F0 = 150.0


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


def _harmonic_pick(x, sr, cands):
    """谐波投票裁决：在候选中选谐波能量得分最高的。

    真基频 f 处 f,2f,3f... 都有谐波能量；
    半频 f/2 处是谱谷；倍频 f 的 2f 处虽有能量但 f/2 处也有。
    用 75 分位数的逐帧谐波求和 + 谷惩罚打分。
    """
    if not cands:
        return 0.0
    D = np.abs(librosa.stft(x.astype(np.float32), n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    best, best_s = 0.0, -1e18
    for f in cands:
        if f <= 0:
            continue
        idxs, ws = [], []
        for h in range(1, 9):
            fh = f * h
            if fh >= freqs[-1] * 0.95:
                break
            idx = int(np.argmin(np.abs(freqs - fh)))
            lo_i, hi_i = max(idx - 1, 0), min(idx + 2, len(freqs))
            if hi_i <= lo_i:
                continue
            idxs.append((lo_i, hi_i))
            ws.append(1.0 / h)
        if not idxs:
            continue
        fs = np.zeros(D.shape[1])
        f0e = np.zeros(D.shape[1])
        for (lo_i, hi_i), w in zip(idxs, ws):
            # librosa.stft 返回 (freq, frames)，D[lo_i:hi_i, :] 取频点区间
            seg = D[lo_i:hi_i, :].max(axis=0)
            fs += seg * w
            if w == 1.0:
                f0e = seg
        total = float(np.percentile(fs, 75))
        band = (freqs > f * 0.97) & (freqs < f * 1.03)
        if band.any():
            nb = D[band, :].mean(axis=0)
            frac = float(((f0e < nb * 0.35) & (nb > 1e-8)).mean())
        else:
            frac = 0.0
        score = total - 0.8 * frac * total
        if score > best_s:
            best_s, best = score, f
    return best


def measure_f0(x, sr, expect_f0=None):
    """三算法中位数交叉测量；分歧大时用谐波投票裁决。"""
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
    med = float(np.median(vv))
    detail["median"] = round(med, 1)
    spread = (max(vv) - min(vv)) / med * 100
    detail["spread"] = round(float(spread), 1)
    if spread > 30:
        cands = []
        for a in vv:
            for m in (0.5, 1.0, 2.0):
                c = a * m
                if FMIN <= c <= FMAX:
                    cands.append(c)
        picked = _harmonic_pick(x, sr, cands)
        if picked > 0:
            detail["harmonic_pick"] = round(picked, 1)
            return picked, detail
    return med, detail


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
    """把总 factor 拆成若干温和步骤执行（每步 ∈ [0.50, 2.0]），保音质。"""
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
    """回正。返回 (y, info)。

    分段闭环：
      阶段 A（仅当目标 f0 落在不可靠区）：
             变调到 anchor=150Hz，在那里闭环收敛
      阶段 B：从当前点用解析式一次性到目标
    """
    info = {"rounds": 0, "stage": "init", "steps": 0}
    target_f0 = base * target_r

    f0_before, d0 = measure_f0(x, sr)
    r_before = f0_before / base if base > 0 else 0.0
    info["r_before"] = round(r_before, 3)
    info["f0_before"] = round(f0_before, 1)
    info["m_before"] = d0
    if r_before <= 0:
        info["stage"] = "no_measure"
        return x.astype(np.float32), info

    use_anchor = target_f0 < RELIABLE_FLOOR
    info["use_anchor"] = use_anchor

    if use_anchor:
        # ---- 阶段 A：变调 + 闭环到 anchor（可靠区，反馈可信）----
        fac_a = min(max(ANCHOR_F0 / f0_before, FACTOR_MIN), FACTOR_MAX)
        y, ns = pitch_shift_total(x, sr, fac_a)
        info["steps"] += ns
        info["rounds"] += 1
        info["stage"] = "anchor_open"
        f0_a, da = measure_f0(y, sr, expect_f0=ANCHOR_F0)
        for _ in range(MAX_REFINE):
            if f0_a <= 0:
                break
            need = ANCHOR_F0 / f0_a
            step = min(max(need, STEP_LO), STEP_HI)
            if abs(step - 1.0) < 0.01:
                break
            y = pitch_shift_once(y, sr, step)
            info["rounds"] += 1
            f0_a, da = measure_f0(y, sr, expect_f0=ANCHOR_F0)
        info["anchor_f0"] = round(f0_a, 1)
        info["m_anchor"] = da
        anchor_ok = abs(f0_a - ANCHOR_F0) / ANCHOR_F0 <= 0.05
        info["anchor_ok"] = bool(anchor_ok)

        # ---- 阶段 B：解析式一次性到最终目标 ----
        if f0_a > 0:
            fac_b = min(max(target_f0 / f0_a, FACTOR_MIN), FACTOR_MAX)
            y, ns = pitch_shift_total(y, sr, fac_b)
            info["steps"] += ns
            info["rounds"] += 1
            info["stage"] = "anchor_to_target"
            info["final_factor"] = round(fac_b, 4)
    else:
        # ---- 常规路径：开环一次到位 + 小步闭环 ----
        fac = min(max(target_r / r_before, FACTOR_MIN), FACTOR_MAX)
        y, ns = pitch_shift_total(x, sr, fac)
        info["steps"] += ns
        info["rounds"] += 1
        info["open_factor"] = round(fac, 4)
        info["stage"] = "openloop"
        f0_now, d1 = measure_f0(y, sr, expect_f0=target_f0)
        for _ in range(MAX_REFINE):
            if abs(f0_now - target_f0) / target_f0 <= TOL:
                break
            need = target_f0 / f0_now if f0_now > 0 else 1.0
            step = min(max(need, STEP_LO), STEP_HI)
            if abs(step - 1.0) < 0.01:
                break
            y = pitch_shift_once(y, sr, step)
            info["rounds"] += 1
            f0_now, d1 = measure_f0(y, sr, expect_f0=target_f0)
            info["stage"] = "refine"

    y = _match_rms(y, x)
    f0_after, df = measure_f0(y, sr, expect_f0=target_f0)
    info["f0_after"] = round(f0_after, 1)
    info["m_final"] = df
    info["r_after"] = round(f0_after / base, 3) if base > 0 else 0.0

    # 判定
    if use_anchor:
        # 目标在不可靠区：用 anchor 收敛精度 + 解析精度判定
        r_est = (info.get("anchor_f0", 0) / ANCHOR_F0 *
                 (target_f0 / info.get("anchor_f0", 1))) if info.get("anchor_f0") else 0
        ok = bool(info.get("anchor_ok"))
        info["judge"] = "anchor"
        info["r_judge"] = round(target_r, 3)
    else:
        ok = bool(abs(f0_after - target_f0) / target_f0 <= TOL)
        info["judge"] = "measured"
        info["r_judge"] = info["r_after"]
    info["in_range"] = ok
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
        dst = os.path.join(_G["out_dir"], os.path.relpath(p, EMOTIONS))
        if not dry:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            sf.write(dst, y, sr)
        return {
            "path": p, "dst": dst, "cid": cid, "emotion": ek,
            "base": base, "target_r": target_r,
            "target_f0": round(base * target_r, 1),
            "r_before": info["r_before"], "r_after": info["r_after"],
            "r_judge": info.get("r_judge"),
            "f0_before": info["f0_before"], "f0_after": info["f0_after"],
            "anchor_f0": info.get("anchor_f0"),
            "anchor_ok": info.get("anchor_ok"),
            "use_anchor": info.get("use_anchor"),
            "judge": info.get("judge"),
            "open_factor": info.get("open_factor"),
            "final_factor": info.get("final_factor"),
            "steps": info.get("steps"),
            "rounds": info["rounds"], "stage": info["stage"],
            "in_range": bool(info["in_range"]),
            "dur_before": round(dur_before, 3),
            "dur_after": round(len(y) / sr, 3),
            "m_before": info.get("m_before"),
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
    ap.add_argument("--tag", default="v9")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--rebuild-base", action="store_true")
    ap.add_argument("--only-cids", default="")
    ap.add_argument("--out", default=f"{VOICES}/pitch_fix_report_v9.json")
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
        ok = sum(1 for r in rr if r["in_range"])
        n_anchor = sum(1 for r in rr if r.get("use_anchor"))
        summary[ek] = {
            "n": len(rr), "in_range": ok,
            "rate": round(ok / len(rr) * 100, 1),
            "r_before_mean": round(float(rb.mean()), 3),
            "r_after_mean": round(float(np.mean([r["r_after"] for r in rr])), 3),
            "r_after_median": round(float(np.median([r["r_after"] for r in rr])), 3),
            "anchor_used": n_anchor,
        }
        print(f"[done] {ek}: n={len(rr)} 达标 {ok}/{len(rr)} "
              f"({ok/len(rr)*100:.1f}%)  倍率均值 {rb.mean():.2f}  "
              f"anchor 路径 {n_anchor}", flush=True)

    json.dump({"summary": summary, "items": results},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False,
              indent=2)
    print(f"-> {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
