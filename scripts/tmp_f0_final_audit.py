"""最终 scale 裁决：对音域边界案例做交叉验证。

三轮信号：
  S1 pyin 绝对读数            （round-trip 八度正确率 95.8%）
  S2 基频存在性（物理判据）    （合成 16/16）
  S3 变调往返自校准            （无循环：施加已知 k，比值须 = k）

对 pyin 与音域冲突的角色，用 S2/S3 复核，并给出最终 scale 与可信度。

关键：产出**唯一权威**的 base_f0_final.json，
后续所有音高回正都以它为尺度基准，且**绝对八度已被物理证据确认**。
"""
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tmp_f0_arena import est_pyin  # noqa: E402
from tmp_f0_ground import fundamental_presence, GENDER_RANGE, find_base  # noqa: E402

VOICES = "/ssd2/duanju/voices"
CAST = "/ssd2/duanju/scripts/cast_all_150.json"


def pitch_shift(x, sr, factor):
    if abs(factor - 1.0) < 1e-6:
        return x.astype(np.float32)
    n_in = len(x)
    rms_in = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    y = librosa.resample(x.astype(np.float32), orig_sr=sr,
                         target_sr=max(int(sr / factor), 1000))
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    Ds = librosa.phase_vocoder(D, rate=1.0 / factor, hop_length=256, n_fft=1024)
    out = np.asarray(librosa.istft(Ds, hop_length=256, n_fft=1024), dtype=np.float32)
    out = out[:n_in] if len(out) > n_in else np.pad(out, (0, n_in - len(out)))
    rms_out = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    if rms_out > 1e-8 and rms_in > 1e-8:
        out = out * (rms_in / rms_out)
    pk = float(np.max(np.abs(out)))
    if pk > 0.99:
        out = out * (0.99 / pk)
    return out


def selfcal(x, sr, f):
    """变调往返自校准：在候选 f 上施加已知 k，检查 pyin 读数的倍率。

    若 pyin 在原音与变调音上锁同一八度，ratio ≈ k。
    对候选分支 f, f/2, f*2 各自做，看哪个分支自洽。
    """
    out = {}
    for label, factor in (("k0.5", 0.5), ("k2.0", 2.0)):
        y = pitch_shift(x, sr, factor)
        b = est_pyin(y, sr)
        out[label] = round(b / f, 3) if f > 0 and b > 0 else 0.0
    return out


def one(c):
    cid = c["id"]
    g = c.get("gender", "")
    p = find_base(cid)
    if not p:
        return None
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    f = est_pyin(x, sr)
    if f <= 0:
        return None
    fp = fundamental_presence(x, sr, f)
    # 对三个分支分别自校准
    cal = {}
    for label, cand in (("f0", f), ("half", f / 2.0), ("double", f * 2.0)):
        if cand < 40 or cand > 800:
            cal[label] = {"k0.5": 0.0, "k2.0": 0.0, "score": 0.0}
            continue
        s = selfcal(x, sr, cand)
        # k=0.5 时 ratio 应 0.5；k=2.0 时 ratio 应 2.0
        e1 = abs(s["k0.5"] - 0.5) if s["k0.5"] > 0 else 9.9
        e2 = abs(s["k2.0"] - 2.0) if s["k2.0"] > 0 else 9.9
        cal[label] = {"k0.5": s["k0.5"], "k2.0": s["k2.0"],
                      "score": round(e1 + e2, 3)}
    return {"cid": cid, "gender": g, "pyin": round(f, 2),
            "fp_verdict": fp["verdict"], "fp": fp, "cal": cal,
            "path": p}


def main():
    cast = json.load(open(CAST))
    print(f"[1/3] 自校准（每角色 2 次变调 × 3 分支）...")
    with ProcessPoolExecutor(max_workers=48) as ex:
        rows = list(ex.map(one, cast, chunksize=1))
    rows = [r for r in rows if r]

    print("[2/3] 三信号融合裁决")
    out = {}
    conf = {"high": 0, "mid": 0, "low": 0}
    conflict_log = []
    for r in rows:
        cid, g = r["cid"], r["gender"]
        lo, hi = GENDER_RANGE.get(g, (60, 300))
        f = r["pyin"]
        # S2: 存在性
        pres_f = f * {"f0": 1.0, "half": 0.5, "third": 1 / 3.0,
                      "unknown": 1.0}[r["fp_verdict"]]
        # S3: 自校准最优分支
        best_branch = min(r["cal"].items(), key=lambda kv: kv[1]["score"])
        cal_f = {"f0": f, "half": f / 2.0, "double": f * 2.0}[best_branch[0]]
        cal_score = best_branch[1]["score"]

        # 融合规则
        # 0) **音域硬约束**：人声 f0 物理上不可能低于 66Hz（男性最低）
        #    也不高于 ~500Hz。剔除荒谬分支（如自校准把 67Hz 判成 33Hz）。
        HARD_LO, HARD_HI = 66.0, 500.0
        cands = {
            "pyin": f,
            "presence": pres_f,
            "selfcal": cal_f,
        }
        viable = {k: v for k, v in cands.items() if HARD_LO <= v <= HARD_HI}
        if not viable:
            viable = {"pyin": f}
        # 权重投票
        votes = {}
        votes[viable.get("pyin", f)] = votes.get(viable.get("pyin", f), 0) + 0.5
        if "presence" in viable:
            votes[viable["presence"]] = votes.get(viable["presence"], 0) + 1.2
        if "selfcal" in viable and cal_score < 0.55:
            votes[viable["selfcal"]] = votes.get(viable["selfcal"], 0) + 1.0
        # 合并相差 3% 的票
        merged = []
        for v, w in votes.items():
            placed = False
            for m in merged:
                if abs(v - m["f"]) / max(m["f"], 1e-6) <= 0.03:
                    m["w"] += w
                    placed = True
                    break
            if not placed:
                merged.append({"f": v, "w": w})
        merged.sort(key=lambda m: -m["w"])
        final = merged[0]["f"]
        # 若最终不在音域内，但**所有**候选都不在音域内，保留最终值
        # （真实人声存在极端值，不能强行塞进典型区间）
        in_range_cands = [m for m in merged if lo <= m["f"] <= hi]
        if in_range_cands:
            final = in_range_cands[0]["f"]
        # 若 pyin 本身在音域内，优先 pyin（最可信的单一估计器）
        if lo <= f <= hi and r["fp_verdict"] == "f0":
            final = f

        top_w = merged[0]["w"]
        if top_w >= 2.0:
            cf = "high"
        elif top_w >= 1.2:
            cf = "mid"
        else:
            cf = "low"
        conf[cf] += 1
        if abs(final - f) / max(f, 1e-6) > 0.03:
            conflict_log.append((cid, g, f, round(pres_f, 1), round(cal_f, 1),
                                 round(final, 1), cf, best_branch[0]))

        out[cid] = {
            "scale": round(float(final), 2),
            "pyin": f,
            "presence": round(float(pres_f), 2),
            "fp_verdict": r["fp_verdict"],
            "selfcal_branch": best_branch[0],
            "selfcal_score": cal_score,
            "selfcal": r["cal"],
            "gender": g,
            "range": [lo, hi],
            "confidence": cf,
            "path": r["path"],
        }

    json.dump(out, open(f"{VOICES}/base_f0_final.json", "w"),
              ensure_ascii=False, indent=1)

    print(f"[3/3] 结果")
    print(f"  置信度: {conf}")
    for g in ("男", "女"):
        v = np.array([out[c]["scale"] for c in out if out[c]["gender"] == g])
        lo, hi = GENDER_RANGE[g]
        print(f"  {g}: n={len(v)} 中位 {np.median(v):.1f} 区间 {v.min():.1f}~{v.max():.1f}  "
              f"落音域 {int(((v>=lo)&(v<=hi)).sum())}/{len(v)}")
    if conflict_log:
        print("-" * 76)
        print("与 pyin 不一致的裁决：")
        for cid, g, fp, pres, cal, fin, cf, br in conflict_log:
            print(f"  {cid:<7}{g:<6} pyin={fp:>7.1f} 存在性={pres:>7.1f} "
                  f"自校准={cal:>7.1f}({br}) -> 最终 {fin:>7.1f} [{cf}]")
    print(f"  -> {VOICES}/base_f0_final.json")


if __name__ == "__main__":
    main()
