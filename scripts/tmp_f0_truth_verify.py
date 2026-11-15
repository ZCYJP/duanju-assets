"""无循环验证：估计器在**真实人声**上的绝对精度与八度正确性。

原理
----
变调算子是精确的（已独立验证 0.1~1.1%）。所以：
    真 f0(变调后) = k × 真 f0(变调前)
对任意真实人声 x 施加**已知**因子 k，然后分别用估计器测量前后：
    ratio = f_est(后) / f_est(前)
若估计器两次都锁在同一八度，ratio ≈ k；
若锁错八度，ratio 会是 k×2、k/2、k×3……
=> ratio ≈ k 的样本占比 = **估计器的八度正确率**，且完全无循环论证。

同时对比多个估计器，选出最可靠者作为最终真值来源。
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tmp_f0_truth2 import measure_truth  # noqa: E402

VOICES = "/ssd2/duanju/voices"


def pitch_shift(x, sr, factor):
    """与生产环境完全一致的变调（复用 v13 的 pitch_shift_once）。"""
    import librosa
    if abs(factor - 1.0) < 1e-6:
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
    else:
        out = out[:n_in]
    rms_out = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    if rms_out > 1e-8 and rms_in > 1e-8:
        out = out * (rms_in / rms_out)
    pk = float(np.max(np.abs(out)))
    if pk > 0.99:
        out = out * (0.99 / pk)
    return out


def one(job):
    path, factor = job
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    r0 = measure_truth(x, sr)
    if r0["f0"] <= 0:
        return None
    y = pitch_shift(x, sr, factor)
    r1 = measure_truth(y, sr)
    if r1["f0"] <= 0:
        return None
    ratio = r1["f0"] / r0["f0"]
    return {"path": path, "factor": factor,
            "f0_before": r0["f0"], "f0_after": r1["f0"],
            "ratio": round(ratio, 4),
            "ratio_rel": round(ratio / factor, 4),
            "err_rel": round(abs(ratio / factor - 1.0), 4),
            "oct_shift": round(np.log2(ratio / factor), 3),
            "lad0": r0.get("ladder_d", 0), "lad1": r1.get("ladder_d", 0),
            "cov0": r0.get("ladder_cov", 0), "cov1": r1.get("ladder_cov", 0),
            "vote0": r0.get("n_vote", 0), "vote1": r1.get("n_vote", 0)}


def main():
    files = []
    for pat in (f"{VOICES}/samples/*_voice.wav",
                f"{VOICES}/samples_ext/*_voice.wav"):
        files += sorted(glob.glob(pat))
    # 取 40 个代表性样本
    rng = np.random.default_rng(7)
    idx = rng.choice(len(files), size=min(40, len(files)), replace=False)
    sel = [files[i] for i in sorted(idx)]
    print(f"[1/3] 采样 {len(sel)} 个真实人声，变调因子集合 {{0.5, 0.7071, 1.4142, 2.0}}")
    jobs = [(p, f) for p in sel for f in (0.5, 0.70710678, 1.41421356, 2.0)]
    print(f"      共 {len(jobs)} 对 (原音, 变调音)")

    print("[2/3] 测量中 ...")
    rows = []
    with ProcessPoolExecutor(max_workers=48) as ex:
        for r in ex.map(one, jobs, chunksize=2):
            if r:
                rows.append(r)

    print("[3/3] 统计")
    arr = np.array([r["ratio_rel"] for r in rows])
    octs = np.array([r["oct_shift"] for r in rows])
    err = np.abs(arr - 1.0)
    ok = err <= 0.12
    print("=" * 76)
    print("round-trip 无循环验证（ratio 应 ≈ factor）")
    print("=" * 76)
    print(f"  样本 {len(rows)}")
    print(f"  |ratio/factor - 1| 中位 {np.median(err)*100:.2f}%  "
          f"p90 {np.percentile(err,90)*100:.2f}%  最大 {err.max()*100:.2f}%")
    print(f"  八度正确率(|ratio/factor-1|<=12%) = {ok.mean()*100:.1f}%  "
          f"({int(ok.sum())}/{len(rows)})")
    oct_hist = {}
    for o in octs:
        k = int(round(o))
        oct_hist[k] = oct_hist.get(k, 0) + 1
    print(f"  八度偏移直方图 (0=正确): {dict(sorted(oct_hist.items()))}")

    # 按因子分组看
    for f in (0.5, 0.70710678, 1.41421356, 2.0):
        sub = [r for r in rows if abs(r["factor"] - f) < 1e-6]
        if not sub:
            continue
        e = np.array([r["err_rel"] for r in sub])
        o = np.array([r["oct_shift"] for r in sub])
        print(f"  factor={f:.4f}: n={len(sub)} 误差中位 {np.median(e)*100:.2f}%  "
              f"八度正确 {(np.abs(o) < 0.5).mean()*100:.1f}%")

    out = f"{VOICES}/f0_truth_roundtrip.json"
    json.dump({"n": len(rows), "oct_ok": float(ok.mean()),
               "err_median": float(np.median(err)),
               "err_p90": float(np.percentile(err, 90)),
               "rows": rows}, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"  -> {out}")

    # 失败样本
    bad = [r for r in rows if not (abs(r["ratio_rel"] - 1) <= 0.12)]
    if bad:
        print("-" * 76)
        print("失败样本（前 15）：")
        for r in bad[:15]:
            print(f"  {os.path.basename(r['path']):<32} k={r['factor']:.3f} "
                  f"{r['f0_before']:7.1f}->{r['f0_after']:7.1f} "
                  f"ratio={r['ratio']:.3f} oct={r['oct_shift']:+.2f} "
                  f"cov {r['cov0']:.2f}/{r['cov1']:.2f}")


if __name__ == "__main__":
    main()
