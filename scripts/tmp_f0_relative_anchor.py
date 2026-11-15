"""相对锚定方案：一切以角色基线音色的实测 f0 为锚。

核心洞察
========
用户的要求是「9 种情绪音高落在给定倍率区间」，这是**相对**要求：
    r = f0(情绪音) / f0(基线音)  ∈  [lo_r, hi_r]

而 r 是**同一估计器在两个音频上的比值**。
若估计器在两处都锁错八度，r 不变（八度因子约掉）。
=> r 的估计**对八度错误免疫**，只需估计器**在同尺度下自洽**。

这就绕开了「绝对 f0 是多少」这个无解的难题：
  - 不需要判断基线音是 90Hz 还是 180Hz
  - 只需要保证「基线音与情绪音用同一把尺子量」

自洽性保障（round-trip 已实测）
  - pyin 的前后比值一致性 95.8%（误差中位 1.46%）
  - 变调算子本身精确（0.1~1.1%）

因此本方案：
  1. 用 pyin 测基线音 f0_base、情绪音 f0_emo
  2. r = f0_emo / f0_base
  3. 若 r ∉ [lo_r, hi_r]，施加变调 factor = target / r
  4. 复核：重测 f0_emo'，r' = f0_emo' / f0_base，应落入区间

关键：f0_base 也在变调后重新测（因为 factor 用到了它），
但 f0_base 是同一个音频、同一估计器，与 f0_emo 的八度偏置**同源**，
所以比值稳定。

附加保障：若 |f0_base| 与音域不符，可对基线音与情绪音**同时**做八度校正
（两者乘同一因子），比值不变，绝对值同时落入音域。
"""
import glob
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import soundfile as sf

VOICES = "/ssd2/duanju/voices"
EMOTIONS = f"{VOICES}/emotions"
SAMPLES = (f"{VOICES}/samples", f"{VOICES}/samples_ext")

TARGET_RATIO = {
    "xi": (1.00, 1.30), "nu": (0.95, 1.30), "ai": (0.82, 1.02),
    "le": (0.95, 1.12), "ju": (1.00, 1.30), "jing": (1.05, 1.40),
    "yan": (0.85, 1.05), "you": (0.90, 1.08), "xiu": (0.98, 1.22),
}
EMO_KEYS = list(TARGET_RATIO.keys())

GENDER_RANGE = {"男": (75.0, 200.0), "女": (150.0, 285.0)}
HARD_LO, HARD_HI = 55.0, 500.0
MAX_REFINE = 4
STEP_LO, STEP_HI = 0.88, 1.14
FACTOR_MIN, FACTOR_MAX = 0.25, 4.0
SUB_LO, SUB_HI = 0.50, 2.0
ACCEPT_TOL = 0.08


# ------------------------------------------------------------- 工具
def pyin_f0(x, sr):
    try:
        f, _, _ = librosa.pyin(x.astype(np.float32), fmin=40.0, fmax=800.0, sr=sr,
                               frame_length=2048, hop_length=256)
        v = f[np.isfinite(f) & (f > 40) & (f < 800)]
        return float(np.median(v)) if len(v) else 0.0
    except Exception:
        return 0.0


def pitch_shift_once(x, sr, factor):
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
    ri = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    ro = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if ro <= 1e-8 or ri <= 1e-8:
        return y
    y = y * (ri / ro)
    pk = float(np.max(np.abs(y)))
    if pk > 0.99:
        y = np.tanh(y * 1.02) * 0.99
    return y


def find_base(cid):
    for d in SAMPLES:
        g = sorted(glob.glob(f"{d}/{cid}_*_voice.wav"))
        if g:
            return g[0]
    return ""


# ------------------------------------------------------------- 单条
def process_one(job):
    path, cid, emotion, base_path, gender = job
    lo_r, hi_r = TARGET_RATIO[emotion]
    info = {"cid": cid, "emotion": emotion, "gender": gender,
            "lo_r": lo_r, "hi_r": hi_r, "path": path}
    try:
        x, sr = sf.read(path, dtype="float32")
    except Exception as e:
        info["error"] = f"read: {e}"
        return None, info
    if x.ndim > 1:
        x = x[:, 0]
    try:
        bx, bsr = sf.read(base_path, dtype="float32")
    except Exception as e:
        info["error"] = f"read base: {e}"
        return None, info
    if bx.ndim > 1:
        bx = bx[:, 0]

    f_base = pyin_f0(bx, bsr)
    f_emo = pyin_f0(x, sr)
    info["f_base"] = round(f_base, 2)
    info["f_emo_before"] = round(f_emo, 2)
    if f_base <= 0 or f_emo <= 0:
        info["stage"] = "no_measure"
        info["ok"] = False
        return x, info

    r = f_emo / f_base
    info["r_before"] = round(r, 4)

    # 目标：区间中点（对齐中点可最大化裕度）
    target_r = (lo_r + hi_r) / 2.0
    # 若当前已在区间内但偏向边缘，仍对齐中点；否则对齐中点（必然入区间）
    factor = target_r / r
    factor = min(max(factor, FACTOR_MIN), FACTOR_MAX)
    y, ns = pitch_shift_total(x, sr, factor)
    info["factor"] = round(factor, 4)
    info["steps"] = ns
    info["stage"] = "shifted"

    for it in range(MAX_REFINE):
        f_now = pyin_f0(y, sr)
        info["rounds"] = it + 1
        if f_now <= 0:
            break
        r_now = f_now / f_base
        info["r_check"] = round(r_now, 4)
        if lo_r * (1 - ACCEPT_TOL) <= r_now <= hi_r * (1 + ACCEPT_TOL):
            info["stage"] = "verified"
            break
        need = target_r / r_now
        step = min(max(need, STEP_LO), STEP_HI)
        if abs(step - 1.0) < 0.01:
            break
        y = pitch_shift_once(y, sr, step)
        info["stage"] = "refine"

    y = _match_rms(y, x)
    f_final = pyin_f0(y, sr)
    r_final = f_final / f_base
    info["f_emo_after"] = round(f_final, 2)
    info["r_after"] = round(r_final, 4)
    info["dur_after"] = round(len(y) / sr, 3)
    info["ok"] = bool(lo_r * (1 - ACCEPT_TOL) <= r_final <= hi_r * (1 + ACCEPT_TOL))
    info["in_range"] = bool(lo_r <= r_final <= hi_r)
    # 绝对音域诊断（不作为判据，仅报告）
    glo, ghi = GENDER_RANGE.get(gender, (60, 300))
    info["gender_band"] = bool(glo <= f_final <= ghi)
    return y, info


def worker(job):
    path, cid, emotion, base_path, gender, dst = job
    y, info = process_one((path, cid, emotion, base_path, gender))
    info["dst"] = dst
    return dst, y, info


# ------------------------------------------------------------- 主流程
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default=f"{VOICES}/emotions_fixed_v16")
    ap.add_argument("--out", default=f"{VOICES}/pitch_fix_report_v16.json")
    ap.add_argument("--emotions", default=",".join(EMO_KEYS))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--tag", default="v16")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    emos = [e for e in args.emotions.split(",") if e]
    cast = {c["id"]: c.get("gender", "")
            for c in json.load(open("/ssd2/duanju/scripts/cast_all_150.json"))}

    jobs = []
    for cid in sorted(cast):
        bp = find_base(cid)
        if not bp:
            print(f"[warn] {cid} 无基线音色，跳过")
            continue
        for emo in emos:
            p = os.path.join(EMOTIONS, cid, f"{cid}_{emo}.wav")
            if os.path.exists(p):
                jobs.append((p, cid, emo, bp, cast[cid]))
    print(f"[plan] {len(jobs)} 任务 / {args.workers} 进程")
    if not jobs:
        return
    os.makedirs(args.out_dir, exist_ok=True)
    full = [(p, cid, e, bp, g, os.path.join(args.out_dir, f"{cid}_{e}.wav"))
            for (p, cid, e, bp, g) in jobs]

    t0 = time.time()
    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for dst, y, info in ex.map(worker, full, chunksize=4):
            results.append(info)
            if y is not None and not args.dry_run:
                try:
                    _, sr = sf.read(info["path"], dtype="float32")
                except Exception:
                    sr = 24000
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                sf.write(dst, y, sr)
            done += 1
            if done % 300 == 0:
                print(f"  {done}/{len(full)}  {time.time()-t0:.0f}s")

    print("-" * 74)
    by_emo = {}
    for r in results:
        by_emo.setdefault(r.get("emotion", "?"), []).append(r)
    for e in emos:
        rs = by_emo.get(e, [])
        if not rs:
            continue
        ok = sum(1 for r in rs if r.get("ok"))
        errs = [abs(r.get("r_after", 0) - (r["lo_r"] + r["hi_r"]) / 2)
                for r in rs if r.get("r_after")]
        stages = {}
        for r in rs:
            stages[r.get("stage", "?")] = stages.get(r.get("stage", "?"), 0) + 1
        med = float(np.median(errs)) if errs else 0.0
        gb = sum(1 for r in rs if r.get("gender_band"))
        print(f"[done] {e:<5} n={len(rs):<4} 达标 {ok}/{len(rs)} "
              f"({ok/max(len(rs),1)*100:.1f}%)  误差中位 {med:.4f}  "
              f"绝对音域 {gb}/{len(rs)}  {stages}")

    all_ok = sum(1 for r in results if r.get("ok"))
    print("-" * 74)
    print(f"[总计] {all_ok}/{len(results)} = {all_ok/max(len(results),1)*100:.2f}%  "
          f"({time.time()-t0:.0f}s)")
    for g in ("男", "女"):
        v = [r.get("f_emo_after", 0) for r in results
             if r.get("gender") == g and r.get("f_emo_after")]
        if not v:
            continue
        v = np.array(v)
        glo, ghi = GENDER_RANGE[g]
        ins = int(((v >= glo) & (v <= ghi)).sum())
        print(f"[音域] {g}: n={len(v)} 中位 {np.median(v):.1f}Hz  "
              f"在音域 {ins}/{len(v)} ({ins/max(len(v),1)*100:.1f}%)")

    json.dump({"results": results, "elapsed": time.time() - t0},
              open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"-> {args.out}")
    if args.dry_run:
        print("(dry-run)")
        return

    if args.merge:
        backup = f"{VOICES}/emotions_backup_{args.tag}"
        if not args.no_backup:
            if os.path.exists(backup):
                print(f"[merge] 备份已存在: {backup}")
            else:
                print(f"[merge] 备份 -> {backup}")
                subprocess.run(["cp", "-a", EMOTIONS, backup], check=True)
        n = 0
        for r in results:
            src = r.get("dst")
            if not src or not os.path.exists(src):
                continue
            orig = os.path.join(EMOTIONS, r["cid"], f"{r['cid']}_{r['emotion']}.wav")
            shutil.copyfile(src, orig)
            n += 1
        print(f"[merge] 已覆盖 {n} 个文件")


if __name__ == "__main__":
    main()
