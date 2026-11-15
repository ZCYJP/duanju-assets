#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""情绪音色音高回正 v5：升调校准测量口径。

【v5 相对 v4 的关键改动】
=========================
v4 用 pYIN 直接测量，但实测在【降调后】的音频上 pYIN 会系统性偏低
（F17 降调 0.60 倍后读 0.881，真实应为 1.125，低 22%）。

v5 引入【升调校准测量】：测量一律走升调方向。
    z   = 变调(x, K=1.6)      # 升调到高位
    r   = pYIN(z) / K / base   # 反推真实倍率

依据 tmp_calib_verify.py（10 条基线主音色 × 6 个已知真实倍率）：

    真实倍率      直接测量中位   直接测量p90   校准测量中位   校准测量p90
    x0.50            4.7%         36.8%         1.2%          5.7%
    x0.60            1.4%          9.3%         1.8%          4.5%
    x0.75            2.3%          8.5%         2.0%          5.4%
    x1.00            0.0%          0.0%         1.4%          4.0%
    x1.40            1.7%          4.1%         2.2%          7.2%
    x2.00            1.7%          4.7%         4.8%         31.3%

    K 扫描（x0.5/0.6/0.75/1.4 四场景）：
      K=1.3  中位1.92% p90 5.69% 最大21.73%
      K=1.6  中位1.81% p90 5.70% 最大10.54%   ← 最优
      K=2.0  中位2.34% p90 7.02% 最大27.75%

关键：pYIN 在【升调后】读数精确（往返验证 x2.6 误差 0.8%），
      在【降调后】读数失真（会锁到低次谐波）。
      所以测量只走升调方向。

【成本】
校准测量比直接测量多一次变调（廉价），pYIN 次数相同
（分母 base 是常数，不需要重复测量原音频）。
实测单条约 3 秒，1350 条约 70 分钟（单进程），可用多进程并行。

用法
----
  python3 tmp_emotion_pitch_fix_v5.py --emotions nu --limit 30 --dry-run
  python3 tmp_emotion_pitch_fix_v5.py --shard 0 --nshards 6 \
      --out-dir /ssd2/duanju/voices/emotions_fixed
  python3 tmp_emotion_pitch_fix_v5.py --merge --out-dir /ssd2/duanju/voices/emotions_fixed
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import time

import numpy as np
import soundfile as sf
import librosa

VOICES = "/ssd2/duanju/voices"
EMOTIONS = f"{VOICES}/emotions"
BASE_F0 = f"{VOICES}/base_f0_pyin.json"
CAST = "/ssd2/duanju/scripts/cast_all_150.json"

FMIN, FMAX = 70.0, 800.0
CALIB_K = 1.6

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

TOL = 0.03          # 判定达标的容差
STEP_LO, STEP_HI = 0.80, 1.25   # 闭环单步限幅
MAX_REFINE = 4      # 闭环最多几轮


def f0_pyin(x, sr):
    try:
        f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=FMIN, fmax=FMAX,
                                sr=sr, frame_length=1024, hop_length=256)
    except Exception:
        return 0.0
    v = f0[(f0 > FMIN) & (f0 < FMAX)]
    return float(np.median(v)) if len(v) else 0.0


def pitch_shift_once(x, sr, factor):
    """单轮变调（factor<1 降调、>1 升调），返回等长波形，含 rms 幅度补偿。"""
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


def measure_ratio(x, sr, base, calib=True):
    """测量音高倍率。

    calib=True  → 升调校准法（升到高位测再反推），全流程默认
    calib=False → 直接 pYIN 测量（仅作交叉核对）
    """
    if not calib:
        f0 = f0_pyin(x, sr)
        return (f0 / base if base > 0 else 0.0), f0
    z = pitch_shift_once(x, sr, CALIB_K)
    f0_hi = f0_pyin(z, sr)
    r = (f0_hi / CALIB_K / base) if base > 0 else 0.0
    return r, f0_hi / CALIB_K


def _match_rms(y, x):
    """最终 rms 对齐 + tanh 软限幅（p95 对齐会触发硬限幅反而更差）。"""
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
    """把情绪音高压到 target_r 倍基线。返回 (y, info)。"""
    info = {"rounds": 0, "stage": "init", "rescued": False}

    r_before, f0_before = measure_ratio(x, sr, base, calib=True)
    info["r_before"] = round(r_before, 3)
    info["f0_before"] = round(f0_before, 1)
    if r_before <= 0:
        info["stage"] = "no_measure"
        return x.astype(np.float32), info

    # ---- 阶段 1：开环一次到位 ----
    factor = min(max(target_r / r_before, 0.30), 3.5)
    y = pitch_shift_once(x, sr, factor)
    info["rounds"] = 1
    info["open_factor"] = round(factor, 4)
    info["stage"] = "openloop"

    r_now, f0_now = measure_ratio(y, sr, base, calib=True)

    # ---- 阶段 2：小步闭环精修（校准测量可信，可小幅迭代）----
    for _ in range(MAX_REFINE):
        if abs(r_now - target_r) <= TOL:
            break
        need = target_r / r_now if r_now > 0 else 1.0
        step = min(max(need, STEP_LO), STEP_HI)
        if abs(step - 1.0) < 0.01:
            break
        y = pitch_shift_once(y, sr, step)
        info["rounds"] += 1
        r_now, f0_now = measure_ratio(y, sr, base, calib=True)
        info["stage"] = "refine"

    # ---- 阶段 3：残留过大的，二次开环 ----
    if abs(r_now - target_r) > 0.10 and info["rounds"] >= 2:
        y2 = pitch_shift_once(x, sr,
                              min(max(target_r / r_before, 0.30), 3.5))
        r2, f02 = measure_ratio(y2, sr, base, calib=True)
        if abs(r2 - target_r) < abs(r_now - target_r):
            y = y2
            r_now, f0_now = r2, f02
            info["rescued"] = True
            info["rounds"] += 1
            info["stage"] = "reopen"

    y = _match_rms(y, x)
    r_final, f0_final = measure_ratio(y, sr, base, calib=True)
    info["r_after"] = round(r_final, 3)
    info["f0_after"] = round(f0_final, 1)
    # 交叉核对：直接测量的读数（仅参考，不用于判定）
    info["r_after_direct"] = round(
        f0_pyin(y, sr) / base if base > 0 else 0.0, 3)
    return y, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotions",
                    default="xi,nu,ai,le,ju,jing,yan,you,xiu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--tag", default="v5")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=-1,
                    help="分片编号（0 起）；-1 表示不分片")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--out", default=f"{VOICES}/pitch_fix_report_v5.json")
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

    base_map = json.load(open(BASE_F0, encoding="utf-8"))
    gender_map = {}
    if os.path.exists(CAST):
        gender_map = {c["id"]: c.get("gender", "") for c in
                      json.load(open(CAST, encoding="utf-8"))}
    emos = [e.strip() for e in args.emotions.split(",") if e.strip()]
    out_dir = args.out_dir or EMOTIONS

    all_report = []
    for ek in emos:
        lo_r, hi_r = TARGET_RATIO.get(ek, (0.9, 1.3))
        target_r = (lo_r + hi_r) / 2.0
        files = sorted(glob.glob(f"{EMOTIONS}/*/*_{ek}.wav"))
        if args.shard >= 0:
            files = [p for i, p in enumerate(files)
                     if i % args.nshards == args.shard]
        if args.limit:
            files = files[:args.limit]
        fixed = in_before = in_after = 0
        rounds_hist = []
        rescued = 0
        report = []
        t0 = time.time()
        for idx, p in enumerate(files):
            cid = os.path.basename(p).split("_")[0]
            base = base_map.get(cid, 0.0)
            if base <= 0:
                continue
            try:
                x, sr = sf.read(p, dtype="float32")
                if x.ndim > 1:
                    x = x[:, 0]
                dur_before = len(x) / sr

                y, info = pitch_fix(x, sr, base, target_r)
                r_before = info["r_before"]
                r_after = info["r_after"]
                ok = bool(lo_r - TOL <= r_after <= hi_r + TOL)
                in_before += int(lo_r - TOL <= r_before <= hi_r + TOL)
                in_after += int(ok)

                dst = os.path.join(out_dir, os.path.relpath(p, EMOTIONS))
                if out_dir != EMOTIONS:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not args.dry_run:
                    sf.write(dst, y, sr)

                if abs(r_before - target_r) > TOL:
                    fixed += 1
                rounds_hist.append(info["rounds"])
                rescued += int(info["rescued"])
                report.append({
                    "path": p, "dst": dst, "cid": cid, "emotion": ek,
                    "gender": gender_map.get(cid, ""),
                    "base": base, "target_r": target_r,
                    "r_before": r_before, "r_after": r_after,
                    "r_after_direct": info.get("r_after_direct"),
                    "f0_before": info["f0_before"], "f0_after": info["f0_after"],
                    "open_factor": info.get("open_factor"),
                    "rounds": info["rounds"], "stage": info["stage"],
                    "rescued": info["rescued"], "in_range": ok,
                    "dur_before": round(dur_before, 3),
                    "dur_after": round(len(y) / sr, 3),
                })
            except Exception as ex:
                report.append({"path": p, "cid": cid, "emotion": ek,
                               "error": f"{type(ex).__name__}: {str(ex)[:150]}"})
            if (idx + 1) % 20 == 0:
                print(f"  [{ek}] {idx+1}/{len(files)}  {time.time()-t0:.0f}s  "
                      f"达标 {in_after}", flush=True)

        rr = [r for r in report if "r_before" in r]
        if rr:
            rb = np.array([r["r_before"] for r in rr])
            ra = np.array([r.get("r_after", r["r_before"]) for r in rr])
            print(f"[done] {ek}: n={len(files)} 达标 {in_before}->{in_after}  "
                  f"倍率均值 {rb.mean():.2f}->{ra.mean():.2f}  "
                  f"轮次 {np.mean(rounds_hist) if rounds_hist else 0:.1f}  "
                  f"救回 {rescued}  {time.time()-t0:.0f}s", flush=True)
        all_report += report
        json.dump(all_report, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
