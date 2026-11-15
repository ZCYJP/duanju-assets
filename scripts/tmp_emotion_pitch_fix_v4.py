#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""情绪音色音高回正 v4（pYIN 统一口径）。

【v4 相对 v3 的关键改动】
=========================
1. 测量口径换为 librosa.pyin（搜索窗 70~800Hz）。
   依据 tmp_f0_roundtrip.py 的真实语音往返验证（12 条基线主音色，
   5 个已知倍率，测得倍率 vs 真实倍率）：

       口径      总体中位误差   p90      x2.6 误差
       ac400        13.3%      131.6%     30.0%
       yin400       18.0%       72.6%     28.7%
       yin800       12.6%       83.7%     10.9%
       pyin800       2.9%       35.5%      0.8%   ← 最优
       yin1200       9.9%       90.7%     11.0%

   pYIN 在【升调】方向近乎精确（x2.0 误差 2.0%、x2.6 误差 0.8%）。
   之前用自相关，口径本身就有 13% 的系统误差，
   基线与回正用同一把「不准的尺子」会把音高修到偏 13% 的位置。

2. 基线同步重算为 pyin800 口径（base_f0_pyin.json）。
   【必须同口径】基线与回正后必须用同一把尺子。
   实测 pyin800 与原自相关基线：中位差 -0.9%、p90 10.2%、最大 53.6%。

3. 回正策略改为【开环一次性 + 小步闭环精修 + 升调校准救回】：
   所有算法在【大幅降调】后测量都会失真（pYIN 最差案例测得 5.22 倍，
   真实 0.5 倍），所以：
     - 第一次修正【绝不依赖降调后的测量】→ 用开环一次到位；
     - 闭环只在【小步】修正（|step-1| ≤ 0.20）时进行，此时测量可靠；
     - 若开环后仍偏差大（>12%），用【升调校准】救回：
       先升调 K=1.6（升调方向测量精确），在高位测得倍率 r_hi，
       反推真实倍率 = r_hi / K，再一次性变调到位。

【为什么必须开环】
自相关在极端漂移上会发散（实测 M65 r 2.84→4.69、M02 3.13→3.76），
根因是搜索上限 400Hz 撞顶 + 谐波误锁。
v3 用「粗降固定步长」也是被动的；v4 用「一次到位」最干净。

用法
----
  python3 tmp_emotion_pitch_fix_v4.py --emotions nu --dry-run
  python3 tmp_emotion_pitch_fix_v4.py --emotions xi,nu,ai,le,ju,jing,yan,you,xiu \
      --out-dir /ssd2/duanju/voices/emotions_fixed
  python3 tmp_emotion_pitch_fix_v4.py --merge --out-dir /ssd2/duanju/voices/emotions_fixed
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

# 各情绪的目标音高倍率区间（相对角色基线）
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

TOL = 0.03                 # 收敛容差（倍率）
RESCUE_TOL = 0.12          # 超过此残差触发升调校准救回
SMALL_STEP = 0.20          # 闭环仅允许 |step-1| <= 该值
CALIB_K = 1.6              # 升调校准的升调倍率


# ------------------------------------------------------------------ 测量
def f0_pyin(x, sr):
    """pYIN 中位基频，搜索窗 70~800Hz。"""
    try:
        f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=FMIN, fmax=FMAX,
                                sr=sr, frame_length=1024, hop_length=256)
    except Exception:
        return 0.0
    v = f0[(f0 > FMIN) & (f0 < FMAX)]
    return float(np.median(v)) if len(v) else 0.0


# ------------------------------------------------------------------ 变调
def pitch_shift_once(x, sr, factor):
    """单轮变调：factor<1 降调、>1 升调；返回等长波形。

    幅度补偿：phase_vocoder + istft 重建的波形 rms 会掉到 0.42 倍
    （≈-7.6dB），必须拉回，否则回正后的音频在混音里会明显偏轻。
    """
    if abs(factor - 1.0) < 0.005:
        return x.astype(np.float32)
    n_in = len(x)
    rms_in = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    y = librosa.resample(x.astype(np.float32), orig_sr=sr,
                         target_sr=max(int(sr / factor), 1000))
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    Ds = librosa.phase_vocoder(D, rate=1.0 / factor, hop_length=256, n_fft=1024)
    out = np.asarray(librosa.istft(Ds, hop_length=256, n_fft=1024),
                     dtype=np.float32)
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
    """最终 rms 对齐 + tanh 软限幅。

    为什么用 rms 而非 p95：相位声码器变调会降低波形相干性，
    峰值相对 rms 下降（crest 2.40→2.12）。按 p95 对齐会触发硬限幅，
    反而更糟（实测 p95 对齐后 p95 掉到 0.81x）。
    """
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


# ------------------------------------------------------------------ 回正
def pitch_fix(x, sr, base, target_r):
    """把情绪音色音高压到 target_r 倍基线。返回 (y, info)。"""
    info = {"stage": "", "rounds": 0, "rescued": False}

    f0_before = f0_pyin(x, sr)
    r_before = f0_before / base if base > 0 else 0.0
    info["f0_before"] = round(f0_before, 1)
    info["r_before"] = round(r_before, 3)

    if r_before <= 0:
        info["stage"] = "no_measure"
        return x.astype(np.float32), info

    # ---------- 阶段 1：开环一次性到位 ----------
    factor = target_r / r_before
    factor = min(max(factor, 0.30), 3.5)
    y = pitch_shift_once(x, sr, factor)
    info["rounds"] = 1
    info["open_factor"] = round(factor, 4)
    info["stage"] = "openloop"

    f0_now = f0_pyin(y, sr)
    r_now = f0_now / base if base > 0 else 0.0

    # ---------- 阶段 2：小步闭环精修 ----------
    for _ in range(4):
        if abs(r_now - target_r) <= TOL:
            break
        need = target_r / r_now if r_now > 0 else 1.0
        if abs(need - 1.0) > SMALL_STEP:
            break
        y = pitch_shift_once(y, sr, need)
        info["rounds"] += 1
        f0_now = f0_pyin(y, sr)
        r_now = f0_now / base if base > 0 else 0.0

    # ---------- 阶段 3：升调校准救回 ----------
    # 大幅降调后测量会失真（pYIN 曾测得 5.22 倍 / 真实 0.5 倍），
    # 所以若残差仍大，改用「先升到高位测量、再一次性变调」：
    # 升调方向测量精确（x2.6 误差 0.8%），是唯一可靠的校准手段。
    if abs(r_now - target_r) > RESCUE_TOL:
        z = pitch_shift_once(y, sr, CALIB_K)
        f0_hi = f0_pyin(z, sr)
        r_hi = f0_hi / base if base > 0 else 0.0
        if r_hi > 0:
            r_true = r_hi / CALIB_K
            if r_true > 0:
                factor2 = target_r / r_true
                factor2 = min(max(factor2, 0.30), 3.5)
                y2 = pitch_shift_once(y, sr, factor2)
                f0_chk = f0_pyin(y2, sr)
                r_chk = f0_chk / base if base > 0 else 0.0
                # 只有当校准结果明显更好时才采纳
                if abs(r_chk - target_r) < abs(r_now - target_r):
                    y = y2
                    r_now = r_chk
                    info["rescued"] = True
                    info["rounds"] += 2
                    info["rescue_r_true"] = round(r_true, 3)
                    info["stage"] = "calib"

    y = _match_rms(y, x)
    f0_after = f0_pyin(y, sr)
    info["f0_after"] = round(f0_after, 1)
    info["r_after"] = round(f0_after / base, 3) if base > 0 else 0.0
    return y, info


# ------------------------------------------------------------------ 基线
def rebuild_base_f0():
    """用 pyin800 口径重算 150 条基线主音色的 f0。"""
    files = sorted(glob.glob(f"{VOICES}/samples/*_voice.wav") +
                   glob.glob(f"{VOICES}/samples_ext/*_voice.wav"))
    out = {}
    t0 = time.time()
    for i, p in enumerate(files):
        cid = os.path.basename(p).split("_")[0]
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        out[cid] = round(f0_pyin(x, sr), 1)
        if (i + 1) % 50 == 0:
            print(f"  base f0 {i+1}/{len(files)}  {time.time()-t0:.0f}s",
                  flush=True)
    json.dump(out, open(BASE_F0, "w", encoding="utf-8"), indent=2)
    print(f"[base] {len(out)} 条基线重算完成 -> {BASE_F0}", flush=True)
    return out


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotions",
                    default="xi,nu,ai,le,ju,jing,yan,you,xiu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--rebuild-base", action="store_true",
                    help="用 pyin800 口径重算 base_f0_pyin.json")
    ap.add_argument("--limit", type=int, default=0,
                    help="每情绪最多处理多少条（0=全部）")
    ap.add_argument("--out", default=f"{VOICES}/pitch_fix_report_v4.json")
    ap.add_argument("--no-backup", action="store_true",
                    help="合并时不备份原目录（默认会备份）")
    ap.add_argument("--tag", default="v4",
                    help="备份目录名后缀")
    args = ap.parse_args()

    if args.rebuild_base:
        rebuild_base_f0()
        if not args.emotions:
            return

    if args.merge:
        src = args.out_dir or f"{VOICES}/emotions_fixed"
        # 合并前先备份原目录（cp -a 保留结构），确保可回滚
        if not args.no_backup:
            bak = f"{VOICES}/emotions_backup_{args.tag or 'v4'}"
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
        if args.limit:
            files = files[:args.limit]
        fixed = in_before = in_after = 0
        rounds_hist = []
        rescued = 0
        report = []
        t0 = time.time()
        for p in files:
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
                    "f0_before": info["f0_before"], "f0_after": info["f0_after"],
                    "open_factor": info.get("open_factor"),
                    "rounds": info["rounds"], "stage": info["stage"],
                    "rescued": info["rescued"],
                    "in_range": ok,
                    "dur_before": round(dur_before, 3),
                    "dur_after": round(len(y) / sr, 3),
                })
            except Exception as ex:
                report.append({"path": p, "cid": cid, "emotion": ek,
                               "error": f"{type(ex).__name__}: {str(ex)[:150]}"})
            if len(report) % 20 == 0:
                print(f"  [{ek}] {len(report)}/{len(files)}  "
                      f"{time.time()-t0:.0f}s  达标 {in_after}", flush=True)

        rr = [r for r in report if "r_before" in r]
        if rr:
            rb = np.array([r["r_before"] for r in rr])
            ra = np.array([r.get("r_after", r["r_before"]) for r in rr])
            print(f"[done] {ek}: n={len(files)} 达标 {in_before}->{in_after}  "
                  f"倍率均值 {rb.mean():.2f}->{ra.mean():.2f}  "
                  f"轮次 {np.mean(rounds_hist) if rounds_hist else 0:.1f}  "
                  f"救回 {rescued}  {time.time()-t0:.0f}s", flush=True)
        all_report += report
        # 报告始终写盘（dry-run 也要留分析依据）
        json.dump(all_report, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

    if args.dry_run:
        print("[dry-run] 未写音频")
    else:
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
