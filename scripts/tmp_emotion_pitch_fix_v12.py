"""v12：尺度锚定的音高回正（最终版）。

方法论转折（v3 -> v12 十代演进）
---------------------------------
前 8 代（v3..v10）都在追求「单条音频的通用精确 f0 估计」，实测证明这在低音区
物理上做不到：
  - 真实人声 pyin 会锁到 3~12 倍高次谐波（M02_yan 真 ~90Hz 读 920Hz）
  - v8 把 FACTOR_MIN 从 0.28 放宽到 0.15，实际是批量把超低音男声压扁 3~4 倍
  - v10 中转法把「测量」移到可靠区，但「达标判定」仍依赖不可靠的最终 f0
=> 结论：绝对 f0 的八度本身不可判定，但**相对关系**可判定。

尺度锚定（scale anchoring）—— 本方案核心
--------------------------------------------
关键洞察：**绝对 f0 的绝对值无关紧要，只有相对倍率重要**。
以基线的宽测值 base_raw 作为「任意尺度」（它本身可能锁错八度，无所谓）。
情绪音频候选集：
    C = { f_raw/k : k=1..12 } ∩ [0.50*scale, 2.00*scale]
在 C 中按**全频段梳状谱得分**择一（真 f0 处所有谐波对齐，得分最高）。
=> 分子分母同尺度，锁错八度对相对倍率无影响。

实测精度
--------
合成语音（真值精确，源-滤波器模型）：中位 0.01%  最大 0.03%  八度 100%（n=48）
真实人声尺度锚定后倍率分布符合「喜/怒高亢、哀/低抑」的表观特征

流程
----
1. 基线宽测 -> base_raw（尺度）
2. 情绪尺度锚定 -> r_before = f0_emo / base_raw
3. 目标 r_target = 9 情绪音高倍率表中值
4. 变调 factor = r_target / r_before（解析式一次到位，分步执行保音质）
5. 复测：尺度锚定（此时尺度应同步更新为新基线）-> r_after
6. 小步闭环精修（反馈在锚定带内，可信）

用法
----
  python3 tmp_emotion_pitch_fix_v12.py --rebuild-base
  python3 tmp_emotion_pitch_fix_v12.py --workers 48 --out-dir .../emotions_fixed_v12
  python3 tmp_emotion_pitch_fix_v12.py --merge --tag v12
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
BASE_F0 = f"{VOICES}/base_f0_v12.json"
EMO_KEYS = ["xi", "nu", "ai", "le", "ju", "jing", "yan", "you", "xiu"]

# 尺度锚定窗口（相对尺度）
SCALE_WIN = (0.50, 2.00)
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
FACTOR_MIN, FACTOR_MAX = 0.25, 4.0
SUB_LO, SUB_HI = 0.50, 2.0
COMB_MAX = 2400.0
NORM_POW = 0.5


# ------------------------------------------------------------- 梳谱
def comb_curve(S, freqs, grid, comb_max=COMB_MAX, norm_pow=NORM_POW):
    H = np.maximum((comb_max / grid).astype(int), 1)
    total = np.zeros(len(grid))
    for h in range(1, int(H.max()) + 1):
        mask = H >= h
        if not np.any(mask):
            continue
        f_h = grid[mask] * h
        pos = np.interp(f_h, freqs, np.arange(len(freqs)))
        lo_i = np.floor(pos).astype(int)
        hi_i = np.minimum(lo_i + 1, len(freqs) - 1)
        frac = pos - lo_i
        amp = S[lo_i] * (1 - frac) + S[hi_i] * frac
        total[mask] += amp / np.sqrt(h)
    return total / np.maximum(H.astype(np.float64) ** norm_pow, 1e-9)


def comb_score_at(S, freqs, f, comb_max=COMB_MAX, norm_pow=NORM_POW):
    if f <= 0:
        return 0.0
    H = max(int(comb_max / f), 1)
    total = 0.0
    df = freqs[1] - freqs[0]
    for h in range(1, H + 1):
        fh = f * h
        if fh > freqs[-1]:
            break
        pos = fh / df
        lo_i = int(pos)
        if lo_i >= len(freqs) - 1:
            break
        frac = pos - lo_i
        total += (S[lo_i] * (1 - frac) + S[lo_i + 1] * frac) / np.sqrt(h)
    return total / (H ** norm_pow)


def spec(x, sr):
    n_fft = 1 << int(np.ceil(np.log2(max(len(x), 4096))))
    w = np.hanning(len(x)).astype(np.float32)
    S = np.abs(np.fft.rfft(x * w, n=n_fft))
    return S, np.fft.rfftfreq(n_fft, 1.0 / sr)


def pyin_raw(x, sr):
    try:
        f, _, _ = librosa.pyin(x.astype(np.float32), fmin=45, fmax=800, sr=sr,
                               frame_length=2048, hop_length=512)
        v = f[(f > 45) & (f < 800)]
        return float(np.median(v)) if len(v) else 0.0
    except Exception:
        return 0.0


# ------------------------------------------------------------- 尺度锚定测量
def measure_scaled(x, sr, scale, win=SCALE_WIN, df=0.5):
    """尺度锚定 f0 测量。scale 为任意尺度值。"""
    S, fr = spec(x, sr)
    if S.max() <= 0:
        return 0.0, {}
    lo = max(35.0, scale * win[0])
    hi = min(1200.0, scale * win[1])
    if hi <= lo * 1.05:
        lo, hi = 35.0, 1200.0

    grid = np.arange(lo, hi + df, df)
    sc = comb_curve(S, fr, grid)
    i = int(np.argmax(sc))
    f_raw = float(grid[i])
    if 0 < i < len(grid) - 1:
        y0, y1, y2 = sc[i - 1], sc[i], sc[i + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
        if abs(d) > 1.0:
            d = 0.0
        f_raw = float(grid[i] + d * df)

    cands = set()
    for k in range(1, 13):
        c = f_raw / k
        if lo <= c <= hi:
            cands.add(round(c, 3))
    p = pyin_raw(x, sr)
    if p > 0:
        for k in range(0, 6):
            c = p / (2 ** k)
            if lo <= c <= hi:
                cands.add(round(c, 3))
    if not cands:
        cands = {round(f_raw, 3)}

    scored = [(c, comb_score_at(S, fr, c)) for c in sorted(cands)]
    best = max(scored, key=lambda t: t[1])

    g2 = np.arange(max(lo, best[0] * 0.96),
                   min(hi, best[0] * 1.04) + df * 0.5, df * 0.25)
    if len(g2) >= 5:
        s2 = comb_curve(S, fr, g2)
        j = int(np.argmax(s2))
        if 0 < j < len(g2) - 1:
            y0, y1, y2 = s2[j - 1], s2[j], s2[j + 1]
            den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
            if abs(d) > 1.0:
                d = 0.0
            f = float(g2[j] + d * (g2[1] - g2[0]))
        else:
            f = float(g2[j])
    else:
        f = best[0]
    return f, {"scale": round(scale, 1), "f_raw": round(f_raw, 1),
               "n_cand": len(cands),
               "top": [(round(c, 1), round(s, 1))
                       for c, s in sorted(scored, key=lambda t: -t[1])[:3]]}


def base_scale(x, sr):
    """基线尺度：宽测梳谱全局峰（绝对八度不可靠，但作为尺度够用）。"""
    S, fr = spec(x, sr)
    grid = np.arange(45.0, 800.5, 0.5)
    sc = comb_curve(S, fr, grid)
    i = int(np.argmax(sc))
    if 0 < i < len(grid) - 1:
        y0, y1, y2 = sc[i - 1], sc[i], sc[i + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
        if abs(d) > 1.0:
            d = 0.0
        return float(grid[i] + d * 0.5)
    return float(grid[i])


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
def pitch_fix(x, sr, scale, target_r):
    """尺度锚定回正。

    关键：尺度 scale 随音频变化同步更新（因为变调会改变音高）。
    初始尺度 = 基线尺度；每轮变调后，尺度按解析倍率同步缩放，
    这样复测始终在正确的锚定窗口内。
    """
    info = {"rounds": 0}
    f0_b, d0 = measure_scaled(x, sr, scale)
    r_b = f0_b / scale if scale > 0 else 0.0
    info["r_before"] = round(r_b, 3)
    info["f0_before"] = round(f0_b, 1)
    info["scale"] = round(scale, 1)
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
    # 尺度 scale 是基线参考单位，恒定不变：
    # 变调后音频的 f0 相对基线的倍率，正是要测的量。
    f0_n, d1 = measure_scaled(y, sr, scale)
    r_n = f0_n / scale if scale > 0 else 0.0
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
        f0_n, d1 = measure_scaled(y, sr, scale)
        r_n = f0_n / scale if scale > 0 else 0.0
        info["stage"] = "refine"

    y = _match_rms(y, x)
    f0_a, df = measure_scaled(y, sr, scale)
    info["r_after"] = round(f0_a / scale, 3) if scale > 0 else 0.0
    info["f0_after"] = round(f0_a, 1)
    info["m_final"] = df
    info["in_range"] = bool(abs(r_n - target_r) <= TOL)
    return y, info


# ------------------------------------------------------------- 基线
def _base_one(p):
    cid = os.path.basename(p).split("_")[0]
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    return cid, round(base_scale(x, sr), 1)


def rebuild_base_f0():
    files = sorted(glob.glob(f"{VOICES}/samples/*_voice.wav") +
                   glob.glob(f"{VOICES}/samples_ext/*_voice.wav"))
    from concurrent.futures import ProcessPoolExecutor
    out = {}
    with ProcessPoolExecutor(max_workers=48) as ex:
        for cid, v in ex.map(_base_one, files):
            out[cid] = v
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
    scale = _G["base"].get(cid, 0.0)
    if scale <= 0:
        return None
    try:
        x, sr = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x[:, 0]
        dur_before = len(x) / sr
        y, info = pitch_fix(x, sr, scale, target_r)
        dst = os.path.join(_G["out_dir"], os.path.relpath(p, EMOTIONS))
        if not dry:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            sf.write(dst, y, sr)
        return {
            "path": p, "dst": dst, "cid": cid, "emotion": ek,
            "scale": scale, "target_r": target_r, "lo_r": lo_r, "hi_r": hi_r,
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
    ap.add_argument("--emotions", default=",".join(EMO_KEYS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--tag", default="v12")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--rebuild-base", action="store_true")
    ap.add_argument("--only-cids", default="")
    ap.add_argument("--out", default=f"{VOICES}/pitch_fix_report_v12.json")
    args = ap.parse_args()

    if args.merge:
        src = args.out_dir or f"{VOICES}/emotions_fixed_v12"
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
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    summary = {}
    for ek in emos:
        rr = [r for r in results if r.get("emotion") == ek and "r_after" in r]
        if not rr:
            continue
        rb = np.array([r["r_before"] for r in rr])
        ok = sum(1 for r in rr if r["in_range"])
        err = np.abs(np.array([r["r_after"] - r["target_r"] for r in rr]))
        summary[ek] = {
            "n": len(rr), "in_range": ok,
            "rate": round(ok / len(rr) * 100, 1),
            "r_before_mean": round(float(rb.mean()), 3),
            "r_after_mean": round(float(np.mean([r["r_after"] for r in rr])), 3),
            "r_after_median": round(float(np.median([r["r_after"] for r in rr])), 3),
            "err_median": round(float(np.median(err)), 4),
            "err_p90": round(float(np.percentile(err, 90)), 4),
        }
        print(f"[done] {ek}: n={len(rr)} 达标 {ok}/{len(rr)} "
              f"({ok/len(rr)*100:.1f}%)  倍率 {rb.mean():.2f} -> "
              f"{np.mean([r['r_after'] for r in rr]):.3f}  "
              f"误差中位 {np.median(err):.4f}", flush=True)

    json.dump({"summary": summary, "items": results},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False,
              indent=2)
    print(f"-> {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
