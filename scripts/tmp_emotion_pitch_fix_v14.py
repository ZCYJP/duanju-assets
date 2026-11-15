"""v14：假设生成 + 窄窗验证的音高回正。

方法论收敛（v3 -> v14）
-------------------------
v12 已把「绝对 f0 估计」换成「尺度锚定」（合成真值精度 0.01%），
但真实人声上仍有 ~9% 未达标。失败机制已被精确定位（V1 打分变体评测）：
  - 宽测梳谱峰往往是对的（如 F19: raw=348Hz 与真值一致）
  - 错在「候选打分择一」这一步把它换成了 2f0 / f0/2 分支
  - 随后精修循环据此把本已正确的音频继续调坏

v14 的决定性洞察
----------------
变调算子是精确的（实测误差 0.1~1.1%），因此：
  变调后音频的目标 f0 是**已知解析值** = scale × target_r
  在这个值 ±15% 的窄窗内测量，**八度歧义在物理上不存在**
=> 宽测只用来「生成假设」，窄窗验证才用来「做判据」。

算法
----
1. 基线尺度 scale（角色级一致性校正八度，见 _fix_scale_octave）
2. 宽测情绪 f0 -> 八度假设集 H = {f_wide × k, k ∈ {1,2,3,1/2,1/3}}
   过滤到 [0.50, 2.20] × scale
3. 对每个假设 h：
     factor = target_r × scale / h
     变调 -> 在窄窗 [expected×0.85, expected×1.18] 内测（无歧义）
     若窄窗内找到显著梳峰 -> 该假设成立，采纳
4. 采纳后，小步闭环精修（每步都用窄窗验证）
5. 达标判据：r_after ∈ 用户给定的 [lo_r, hi_r] 区间内

用法
----
  python3 tmp_emotion_pitch_fix_v14.py --rebuild-base
  python3 tmp_emotion_pitch_fix_v14.py --workers 48 --out-dir .../emotions_fixed_v14
  python3 tmp_emotion_pitch_fix_v14.py --merge --tag v14
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
BASE_F0 = f"{VOICES}/base_f0_v14.json"
EMO_KEYS = ["xi", "nu", "ai", "le", "ju", "jing", "yan", "you", "xiu"]

# 用户给定的 9 情绪音高倍率区间
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

COMB_MAX = 2400.0
NORM_POW = 0.5
# 假设集窗口（相对基线尺度）
HYP_LO, HYP_HI = 0.50, 2.20
# 窄窗验证窗口（相对 expected）
NARROW_LO, NARROW_HI = 0.85, 1.18
# 窄窗内梳峰显著阈值（相对窄窗外邻近带）
NARROW_PROM = 1.8
# 精修
STEP_LO, STEP_HI = 0.88, 1.14
MAX_REFINE = 3
FACTOR_MIN, FACTOR_MAX = 0.25, 4.0
SUB_LO, SUB_HI = 0.50, 2.0


# ------------------------------------------------------------- 频谱与梳谱
def spec(x, sr):
    n_fft = 1 << int(np.ceil(np.log2(max(len(x), 4096))))
    w = np.hanning(len(x)).astype(np.float32)
    S = np.abs(np.fft.rfft(x * w, n=n_fft))
    return S, np.fft.rfftfreq(n_fft, 1.0 / sr)


def amp_at(S, freqs, f):
    if f <= 0 or f > freqs[-1]:
        return 0.0
    df = freqs[1] - freqs[0]
    pos = f / df
    i = int(pos)
    if i >= len(freqs) - 1:
        return float(S[-1])
    frac = pos - i
    return float(S[i] * (1 - frac) + S[i + 1] * frac)


def comb_curve(S, freqs, grid, comb_max=COMB_MAX, norm_pow=NORM_POW):
    H = np.maximum((comb_max / grid).astype(int), 1)
    total = np.zeros(len(grid))
    for h in range(1, int(H.max()) + 1):
        mask = H >= h
        if not np.any(mask):
            continue
        f_h = grid[mask] * h
        df = freqs[1] - freqs[0]
        pos = f_h / df
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
    for h in range(1, H + 1):
        fh = f * h
        if fh > freqs[-1]:
            break
        total += amp_at(S, freqs, fh) / np.sqrt(h)
    return total / (H ** norm_pow)


def pyin_f0(x, sr):
    try:
        f, _, _ = librosa.pyin(x.astype(np.float32), fmin=45, fmax=800, sr=sr,
                               frame_length=2048, hop_length=512)
        v = f[(f > 45) & (f < 800)]
        return float(np.median(v)) if len(v) else 0.0
    except Exception:
        return 0.0


def wide_measure(x, sr, lo=45.0, hi=800.0, df=0.5):
    """宽测：梳谱全局峰（可能有八度错）。"""
    S, fr = spec(x, sr)
    grid = np.arange(lo, hi + df, df)
    sc = comb_curve(S, fr, grid)
    i = int(np.argmax(sc))
    f = float(grid[i])
    if 0 < i < len(grid) - 1:
        y0, y1, y2 = sc[i - 1], sc[i], sc[i + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
        if abs(d) > 1.0:
            d = 0.0
        f = float(grid[i] + d * df)
    return f, S, fr


def narrow_measure(S, freqs, expected, lo_rel=NARROW_LO, hi_rel=NARROW_HI):
    """窄窗测量：expected 已知 ±15% 内无八度歧义。

    返回 (f0, 显著度)。显著度 = 窗内峰值 / 窗外邻近带中位数。
    """
    lo = expected * lo_rel
    hi = expected * hi_rel
    if hi <= lo * 1.02:
        return 0.0, 0.0
    grid = np.arange(lo, hi + 0.25, 0.25)
    if len(grid) < 5:
        return 0.0, 0.0
    sc = comb_curve(S, freqs, grid)
    i = int(np.argmax(sc))
    peak = float(sc[i])
    # 窗外参照带（低 4 倍窗宽）
    ref_lo = max(35.0, expected * 0.45)
    ref_hi = max(ref_lo + 10.0, expected * 0.75)
    ref_grid = np.arange(ref_lo, ref_hi + 0.5, 0.5)
    ref = comb_curve(S, freqs, ref_grid) if len(ref_grid) >= 5 else sc
    base = float(np.median(ref)) + 1e-12
    prom = peak / base
    if 0 < i < len(grid) - 1:
        y0, y1, y2 = sc[i - 1], sc[i], sc[i + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if abs(den) > 1e-15 else 0.0
        if abs(d) > 1.0:
            d = 0.0
        f = float(grid[i] + d * 0.25)
    else:
        f = float(grid[i])
    return f, prom


# ------------------------------------------------------------- 变调
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
def pitch_fix(x, sr, scale, target_r, lo_r, hi_r):
    info = {"rounds": 0, "scale": round(scale, 1)}
    f_wide, S, fr = wide_measure(x, sr)
    pf = pyin_f0(x, sr)
    info["f_wide"] = round(f_wide, 1)
    info["pyin"] = round(pf, 1)
    if f_wide <= 0:
        info["stage"] = "no_measure"
        return x.astype(np.float32), info

    # 假设集
    hyps = []
    for base_f in (f_wide, pf):
        if base_f <= 0:
            continue
        for k in (1.0, 2.0, 3.0, 0.5, 1.0 / 3.0):
            h = base_f * k
            if HYP_LO * scale <= h <= HYP_HI * scale:
                hyps.append(round(h, 2))
    if not hyps:
        hyps = [round(f_wide, 2)]
    # 去重并按「离 scale 最近」排序（最可能的先试）
    hyps = sorted(set(hyps), key=lambda h: abs(np.log(h / scale)))
    info["hyps"] = hyps[:8]

    expected = scale * target_r
    best = None
    tried = []
    for h in hyps[:6]:
        factor = min(max(expected / h, FACTOR_MIN), FACTOR_MAX)
        y_try, _ = pitch_shift_total(x, sr, factor)
        S2, fr2 = spec(y_try, sr)
        f_n, prom = narrow_measure(S2, fr2, expected)
        rec = {"hyp": h, "factor": round(factor, 4), "f_n": round(f_n, 1),
               "prom": round(prom, 2)}
        tried.append(rec)
        if f_n > 0 and prom >= NARROW_PROM:
            # 采纳第一个成立且最接近目标的
            r_n = f_n / scale
            score = abs(r_n - target_r)
            if best is None or score < best[0]:
                best = (score, h, factor, y_try, f_n, prom, r_n)
            if score <= 0.04:
                break
    info["tries"] = tried

    if best is None:
        # 无假设通过窄窗验证：取「窄窗显著度最高」的假设尽力而为，
        # 随后同样进入精修（窄窗测量在 expected 附近无八度歧义）
        pool = [t for t in tried if t["f_n"] > 0]
        if pool:
            t = max(pool, key=lambda z: z["prom"])
            h = t["hyp"]
            factor = min(max(expected / h, FACTOR_MIN), FACTOR_MAX)
            y, ns = pitch_shift_total(x, sr, factor)
            r_n = t["f_n"] / scale
            info["prom"] = t["prom"]
        else:
            h = hyps[0]
            factor = min(max(expected / h, FACTOR_MIN), FACTOR_MAX)
            y, ns = pitch_shift_total(x, sr, factor)
            r_n = 0.0
        info["stage"] = "fallback"
        info["steps"] = ns
        info["rounds"] = 1
        info["open_factor"] = round(factor, 4)
        info["hyp"] = h
    else:
        _, h, factor, y, f_n, prom, r_n = best
        info["stage"] = "verified"
        info["hyp"] = h
        info["open_factor"] = round(factor, 4)
        info["steps"] = 1
        info["rounds"] = 1
        info["r_after_open"] = round(r_n, 3)
        info["prom"] = round(prom, 2)

    # 精修（统一流程：窄窗验证优先，失败则退回宽测）
    for _ in range(MAX_REFINE):
        if r_n > 0 and lo_r <= r_n <= hi_r and abs(r_n - target_r) <= 0.05:
            break
        need = target_r / r_n if r_n > 0 else 1.0
        step = min(max(need, STEP_LO), STEP_HI)
        if abs(step - 1.0) < 0.01:
            break
        y = pitch_shift_once(y, sr, step)
        info["rounds"] += 1
        S2, fr2 = spec(y, sr)
        f_n2, prom2 = narrow_measure(S2, fr2, expected)
        if f_n2 > 0 and prom2 >= NARROW_PROM:
            r_new = f_n2 / scale
        else:
            f_w2, _, _ = wide_measure(y, sr)
            r_new = f_w2 / scale
        if r_new <= 0:
            break
        r_n = r_new
        if info["stage"] != "fallback":
            info["stage"] = "refine"

    y = _match_rms(y, x)
    S2, fr2 = spec(y, sr)
    f_a, prom = narrow_measure(S2, fr2, expected)
    if f_a > 0 and prom >= NARROW_PROM:
        r_a = f_a / scale
    else:
        f_w2, _, _ = wide_measure(y, sr)
        r_a = f_w2 / scale
    info["r_after"] = round(r_a, 3)
    info["f0_after"] = round(f_a, 1)
    info["prom_final"] = round(prom, 2)
    info["in_range"] = bool(lo_r <= r_a <= hi_r)
    return y, info


# ------------------------------------------------------------- 基线尺度
def _base_one(p):
    cid = os.path.basename(p).split("_")[0]
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    f, _, _ = wide_measure(x, sr)
    return cid, round(f, 1)


def rebuild_base_f0():
    """基线尺度 + 角色级八度一致性校正。

    对每个角色，尝试基线尺度的八度变体 {s, s/2, s*2, s/3, s*3}，
    选使「该角色 9 条情绪的宽测值落在 [0.5,2.2]×尺度 内」的数目最多的那个。
    """
    files = sorted(glob.glob(f"{VOICES}/samples/*_voice.wav") +
                   glob.glob(f"{VOICES}/samples_ext/*_voice.wav"))
    with ProcessPoolExecutor(max_workers=48) as ex:
        base_raw = dict(ex.map(_base_one, files))

    # 收集各角色情绪宽测值
    emo_wide = {}
    for cid in base_raw:
        vals = []
        for ek in EMO_KEYS:
            ep = glob.glob(f"{EMOTIONS}/{cid}/*_{ek}.wav")
            if not ep:
                continue
            x, sr = sf.read(ep[0], dtype="float32")
            if x.ndim > 1:
                x = x[:, 0]
            f, _, _ = wide_measure(x, sr)
            vals.append(f)
        emo_wide[cid] = vals

    out = {}
    fixed = 0
    for cid, s0 in base_raw.items():
        vals = emo_wide.get(cid, [])
        cands = [s0, s0 * 2, s0 / 2, s0 * 3, s0 / 3]
        best, bscore = s0, -1
        for c in cands:
            if c <= 0:
                continue
            n = sum(1 for v in vals if HYP_LO * c <= v <= HYP_HI * c)
            # 距离中位更近者优先（并列时）
            if n > bscore:
                bscore, best = n, c
        out[cid] = round(best, 1)
        if abs(best - s0) > 1.0:
            fixed += 1
    json.dump(out, open(BASE_F0, "w", encoding="utf-8"), indent=2)
    print(f"[base] {len(out)} 条 -> {BASE_F0}（八度校正 {fixed} 个角色）",
          flush=True)
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
        y, info = pitch_fix(x, sr, scale, target_r, lo_r, hi_r)
        dst = os.path.join(_G["out_dir"], os.path.relpath(p, EMOTIONS))
        if not dry:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            sf.write(dst, y, sr)
        rec = {
            "path": p, "dst": dst, "cid": cid, "emotion": ek,
            "scale": scale, "target_r": target_r, "lo_r": lo_r, "hi_r": hi_r,
            "r_before": info.get("r_after_open", 0.0),
            "r_after": info.get("r_after", 0.0),
            "f0_after": info.get("f0_after", 0.0),
            "f_wide": info.get("f_wide"), "pyin": info.get("pyin"),
            "hyp": info.get("hyp"), "open_factor": info.get("open_factor"),
            "prom": info.get("prom"), "prom_final": info.get("prom_final"),
            "stage": info.get("stage"), "rounds": info.get("rounds"),
            "in_range": bool(info.get("in_range")),
            "dur_before": round(dur_before, 3),
            "dur_after": round(len(y) / sr, 3),
        }
        return rec
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
    ap.add_argument("--tag", default="v14")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--rebuild-base", action="store_true")
    ap.add_argument("--only-cids", default="")
    ap.add_argument("--out", default=f"{VOICES}/pitch_fix_report_v14.json")
    args = ap.parse_args()

    if args.merge:
        src = args.out_dir or f"{VOICES}/emotions_fixed_v14"
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
        ok = sum(1 for r in rr if r["in_range"])
        err = np.abs(np.array([r["r_after"] - r["target_r"] for r in rr]))
        stages = {}
        for r in rr:
            stages[r.get("stage", "?")] = stages.get(r.get("stage", "?"), 0) + 1
        summary[ek] = {
            "n": len(rr), "in_range": ok,
            "rate": round(ok / len(rr) * 100, 1),
            "err_median": round(float(np.median(err)), 4),
            "err_p90": round(float(np.percentile(err, 90)), 4),
            "stages": stages,
        }
        print(f"[done] {ek}: n={len(rr)} 达标 {ok}/{len(rr)} "
              f"({ok/len(rr)*100:.1f}%)  误差中位 {np.median(err):.4f}  "
              f"{stages}", flush=True)

    json.dump({"summary": summary, "items": results},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False,
              indent=2)
    print(f"-> {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
