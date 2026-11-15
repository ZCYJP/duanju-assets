#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""情绪音色音高回正 v6：谐波候选投票测量。

【v6 要解决的三个问题】
=======================
问题 1：口径不一致（v5 的隐性 bug）
    v5 用【升调校准】测量情绪音频，但基线 base_f0_pyin.json 是【直接 pYIN】
    测的。F17 上两者差 14%（calib/direct=0.859）-> 倍率全错。
    => v6 要求基线与测量使用完全相同的口径函数。

问题 2：半频/倍频误锁（v5 的硬伤）
    F17_nu 连续降调实测（tmp_f17_diag.py）：
        累计因子 0.504 -> 理论 156.6Hz，实测 direct 159.0Hz（吻合）
    但单次开环降调 0.698 倍后，实测 r=0.735 而预期 r=1.125。
    pYIN 在该音频上锁到了半频（读出约一半的实际基频）。
    => 单一算法（不管 ac / yin / pyin）都有这个问题，
       因为它们都只给一个数，无法发现"这个数是谐波"。

问题 3：单点测量无法自证
    => v6 引入【谐波候选集 + 投票】：
       1) 用多种算法（ac / pyin / hpss+pyin）在同一音频上各给一个候选；
       2) 把每个候选的 1/2、1、2、3 倍都纳入候选集；
       3) 用【谐波能量验证】给每个候选打分：
          在候选频率 f 及其整数倍处取 STFT 幅度，求和；
          真正的基频应该让 1x,2x,3x... 都有能量；
          半频会让 0.5x 处（无谐波）也得分虚高 -> 惩罚；
       4) 取最高分候选。

【谐波能量验证原理】
=====================
对候选 f，在长时谱上计算 S(f), S(2f), S(3f), ...
  - 若 f 是真基频：这些点都落在谐波上 -> 总和最大
  - 若 f 是半频（实际 2f）：f 处通常无谐波能量（谷），
    而 2f,4f,6f 有能量 -> 总和小，且 f 处是谷
  - 若 f 是倍频（实际 f/2）：f,2f 处有能量（都是偶次谐波），
    但 f/2 也有能量 -> 用 "基频处是否有能量" 区分
用「谐波求和 / 全谐波最大值」这类经典打分可以有效区分。

用法
----
  python3 tmp_emotion_pitch_fix_v6.py --emotions nu --limit 30 --dry-run
  python3 tmp_emotion_pitch_fix_v6.py --shard 0 --nshards 8 \
      --out-dir /ssd2/duanju/voices/emotions_fixed
  python3 tmp_emotion_pitch_fix_v6.py --merge
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
BASE_F0 = f"{VOICES}/base_f0_v6.json"
CAST = "/ssd2/duanju/scripts/cast_all_150.json"

FMIN, FMAX = 65.0, 700.0

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
MAX_REFINE = 5


# ---------------------------------------------------------------- 测量层
def _ac_f0(x, sr, fmin=FMIN, fmax=FMAX):
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


def _pyin_f0(x, sr, fmin=FMIN, fmax=FMAX):
    try:
        f0, _, _ = librosa.pyin(x.astype(np.float32), fmin=fmin, fmax=fmax,
                                sr=sr, frame_length=1024, hop_length=256)
    except Exception:
        return 0.0
    v = f0[(f0 > fmin) & (f0 < fmax)]
    return float(np.median(v)) if len(v) else 0.0


def _hpss_pyin_f0(x, sr, fmin=FMIN, fmax=FMAX):
    try:
        yh = librosa.effects.harmonic(x.astype(np.float32))
        f0, _, _ = librosa.pyin(yh, fmin=fmin, fmax=fmax, sr=sr,
                                frame_length=1024, hop_length=256)
    except Exception:
        return 0.0
    v = f0[(f0 > fmin) & (f0 < fmax)]
    return float(np.median(v)) if len(v) else 0.0


def _spectrum(x, sr):
    """长时平均幅度谱（用于谐波能量验证）。"""
    D = np.abs(librosa.stft(x.astype(np.float32), n_fft=2048, hop_length=512))
    # 聚成 [频点, 帧] 的平均谱，便于按频率取值
    S = D.mean(axis=1)
    return S, librosa.fft_frequencies(sr=sr, n_fft=2048)


def _harmonic_score(D, freqs, f):
    """谐波求和得分。

    对候选 f，取 f, 2f, 3f, ...（最多 8 次）处的谱幅度，
    并做 1/h 权重（低次谐波能量更大）。
    同时惩罚"基频处本身是谱谷"的情况：
    若 S(f) 明显低于其邻居平均，说明 f 不是真正的基频。
    """
    total = 0.0
    n_h = 0
    f0_energy = 0.0
    for h in range(1, 9):
        fh = f * h
        if fh >= freqs[-1] * 0.95:
            break
        idx = int(np.argmin(np.abs(freqs - fh)))
        if idx >= len(freqs):
            break
        # 取 ±1 bin 的最大值，容忍频率量化误差
        lo_i = max(idx - 1, 0)
        hi_i = min(idx + 2, len(freqs))
        e = float(np.max(D[lo_i:hi_i])) if hi_i > lo_i else 0.0
        total += e / h
        n_h += 1
        if h == 1:
            f0_energy = e
    if n_h == 0:
        return -1.0, 0.0
    # 邻域基线：f 附近 ±3% 的平均能量
    band = (freqs > f * 0.97) & (freqs < f * 1.03)
    nb = float(D[band].mean()) if band.any() else 0.0
    # 谷惩罚：基频处能量低于邻域的 35% 时重罚
    valley_pen = 0.0
    if nb > 1e-8 and f0_energy < nb * 0.35:
        valley_pen = -0.8 * (1.0 - f0_energy / (nb * 0.35))
    return total, valley_pen


def measure_f0(x, sr, base=None, calib=True):
    """返回 (f0, detail)。

    谐波候选投票：
      1) 三种算法各给一个候选
      2) 每个候选的 1/2, 1/3, 1, 2, 3 倍都进候选池
      3) 谐波能量 + 谷惩罚打分
      4) 最高分胜出
    calib=True 时先升调 K 倍再测（升调方向测量更可靠）。
    """
    detail = {}
    src = x
    if calib:
        src = pitch_shift_once(x, sr, CALIB_K)

    cands_raw = []
    for name, fn in (("ac", _ac_f0), ("pyin", _pyin_f0),
                     ("hpss_pyin", _hpss_pyin_f0)):
        v = fn(src, sr)
        detail[name] = round(v, 1)
        if v > 0:
            cands_raw.append(v)

    if not cands_raw:
        return 0.0, detail

    D, freqs = _spectrum(src, sr)
    pool = set()
    for v in cands_raw:
        for m in (0.5, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.5, 2.0, 3.0):
            fc = v * m
            if FMIN <= fc <= FMAX:
                pool.add(round(fc, 2))

    scored = []
    for fc in pool:
        s, pen = _harmonic_score(D, freqs, fc)
        scored.append((s + pen, fc))
    scored.sort(reverse=True)
    detail["pool"] = [f"{f:.1f}" for _, f in scored[:6]]
    detail["top_score"] = round(scored[0][0], 4) if scored else 0.0
    if not scored or scored[0][0] <= 0:
        return 0.0, detail
    best_f = scored[0][1]
    detail["chosen"] = round(best_f, 1)

    if calib:
        best_f = best_f / CALIB_K
    return best_f, detail


# ---------------------------------------------------------------- 变调层
CALIB_K = 1.6


def pitch_shift_once(x, sr, factor):
    """单轮变调（factor<1 降调、>1 升调），等长输出 + rms 幅度补偿。"""
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


def ratio_of(x, sr, base):
    f0, detail = measure_f0(x, sr, base, calib=True)
    return (f0 / base if base > 0 else 0.0), f0, detail


def pitch_fix(x, sr, base, target_r):
    info = {"rounds": 0, "stage": "init", "rescued": False}
    r_before, f0_before, d0 = ratio_of(x, sr, base)
    info["r_before"] = round(r_before, 3)
    info["f0_before"] = round(f0_before, 1)
    info["m_before"] = d0
    if r_before <= 0:
        info["stage"] = "no_measure"
        return x.astype(np.float32), info

    # 阶段 1：开环一次到位
    factor = min(max(target_r / r_before, 0.30), 3.5)
    y = pitch_shift_once(x, sr, factor)
    info["rounds"] = 1
    info["open_factor"] = round(factor, 4)
    info["stage"] = "openloop"
    r_now, f0_now, d1 = ratio_of(y, sr, base)
    info["m_after"] = d1

    # 阶段 2：小步闭环精修
    for _ in range(MAX_REFINE):
        if abs(r_now - target_r) <= TOL:
            break
        need = target_r / r_now if r_now > 0 else 1.0
        step = min(max(need, 0.82), 1.22)
        if abs(step - 1.0) < 0.01:
            break
        y = pitch_shift_once(y, sr, step)
        info["rounds"] += 1
        r_now, f0_now, d1 = ratio_of(y, sr, base)
        info["stage"] = "refine"

    # 阶段 3：残留大 -> 用候选池里更优的谐波候选重新定向
    if abs(r_now - target_r) > 0.08:
        # 尝试：把开环因子按候选池里第二/第三候选重算
        cands = d1.get("pool") or []
        tried = False
        for cs in cands[:3]:
            try:
                fc = float(cs) / CALIB_K
            except ValueError:
                continue
            if fc <= 0:
                continue
            r_alt = fc / base if base > 0 else 0.0
            if r_alt <= 0:
                continue
            factor2 = min(max(target_r / r_alt, 0.30), 3.5)
            y2 = pitch_shift_once(x, sr, factor2)
            r2, _, _ = ratio_of(y2, sr, base)
            if abs(r2 - target_r) < abs(r_now - target_r):
                y = y2
                r_now = r2
                info["rescued"] = True
                info["rounds"] += 1
                info["stage"] = "repool"
                tried = True
                break
        if not tried:
            info["stage"] = "openloop" if info["stage"] == "init" else info["stage"]

    y = _match_rms(y, x)
    r_final, f0_final, df = ratio_of(y, sr, base)
    info["r_after"] = round(r_final, 3)
    info["f0_after"] = round(f0_final, 1)
    info["m_final"] = df
    return y, info


# ---------------------------------------------------------------- 基线
def rebuild_base_f0():
    """用与 measure_f0 完全相同的口径（calib + 谐波投票）重算基线。"""
    files = sorted(glob.glob(f"{VOICES}/samples/*_voice.wav") +
                   glob.glob(f"{VOICES}/samples_ext/*_voice.wav"))
    out = {}
    t0 = time.time()
    for i, p in enumerate(files):
        cid = os.path.basename(p).split("_")[0]
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        f0, d = measure_f0(x, sr, None, calib=True)
        out[cid] = round(f0, 1)
        if (i + 1) % 25 == 0:
            print(f"  base {i+1}/{len(files)}  {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(BASE_F0, "w", encoding="utf-8"), indent=2)
    print(f"[base] {len(out)} 条 -> {BASE_F0}", flush=True)
    return out


# ---------------------------------------------------------------- 并行
_G = {}


def _init_worker(out_dir, base_path):
    _G["out_dir"] = out_dir
    _G["base"] = json.load(open(base_path, encoding="utf-8"))


def _work(job):
    ek, p, target_r, lo_r, hi_r = job
    base = _G["base"].get(os.path.basename(p).split("_")[0], 0.0)
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
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        sf.write(dst, y, sr)
        return {
            "path": p, "dst": dst,
            "cid": os.path.basename(p).split("_")[0], "emotion": ek,
            "base": base, "target_r": target_r,
            "r_before": info["r_before"], "r_after": r_after,
            "f0_before": info["f0_before"], "f0_after": info["f0_after"],
            "open_factor": info.get("open_factor"),
            "rounds": info["rounds"], "stage": info["stage"],
            "rescued": info["rescued"], "in_range": ok,
            "dur_before": round(dur_before, 3),
            "dur_after": round(len(y) / sr, 3),
            "m_before": info.get("m_before"),
            "m_final": info.get("m_final"),
        }
    except Exception as ex:
        return {"path": p, "cid": os.path.basename(p).split("_")[0],
                "emotion": ek,
                "error": f"{type(ex).__name__}: {str(ex)[:200]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotions",
                    default="xi,nu,ai,le,ju,jing,yan,you,xiu")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--tag", default="v6")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rebuild-base", action="store_true")
    ap.add_argument("--out", default=f"{VOICES}/pitch_fix_report_v6.json")
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
            jobs.append((ek, p, target_r, lo_r, hi_r))

    print(f"[plan] {len(jobs)} 个任务，{args.workers} 进程并行", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(out_dir, BASE_F0)) as ex:
        for i, r in enumerate(ex.map(_work, jobs, chunksize=2)):
            if r:
                results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    # 统计
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
            "rescued": sum(1 for r in rr if r["rescued"]),
        }
        print(f"[done] {ek}: n={len(rr)} 达标 {ok}/{len(rr)} "
              f"({ok/len(rr)*100:.1f}%)  倍率 {rb.mean():.2f}->{ra.mean():.2f}",
              flush=True)

    json.dump({"summary": summary, "items": results},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False,
              indent=2)
    print(f"-> {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
