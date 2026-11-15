#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""100 种动物角色 × 3 种情绪声音（喜/怒/哀）。

链路复用 gen_emotion_voices_v2.py：Qwen3-TTS VoiceDesign + 多候选采样 + 声学指标选优。

与人物版的差异：
  * 动物没有预设基线音高 → 先为每个角色生成一条中性基线音频，测出 f0_mean 作为基线，
    后续 3 种情绪的音高一律用「相对基线的倍率」判定，避免音高漂移。
  * 抖动 f0_std 也改成相对基线的倍率（动物基线音高比人高，绝对阈值不适用）。
  * 输出目录 /ssd2/duanju/voices/animals/<ID>/<ID>_<emo>.wav

用法：
  python3 gen_animals_voices.py --shard 0 --of 7 --gpu 0 --cand 3
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

OUT = "/ssd2/duanju/voices/animals"
DESIGN = os.environ.get("QWEN_TTS_DESIGN", "/ssd2/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
ANIMALS = os.environ.get("ANIMALS_JSON", "/ssd2/duanju/scripts/animals_100.json")
BASE_F0_DIR = "/ssd2/duanju/voices/base_f0_animals"

# 3 种情绪：key, 中文, 台词, 情绪描述
EMOS = {
    "xi": ("喜",
           "太好了！哈哈，这下可让我高兴坏了，真是让人忍不住笑出声来！",
           "非常开心的声音。音调比平时略高一点，整体语调上扬、结尾带兴奋的高音，"
           "但不要尖锐刺耳。节奏轻快跳跃、富有弹性，音量饱满响亮，"
           "笑声明脆爽朗带着欢呼感，气息充沛不喘。"),
    "nu": ("怒",
           "够了！你别再说了，这件事我绝不会让步——滚出去！",
           "非常愤怒的声音。音量极大、几乎在吼，重音明显、字字砸下来，"
           "语速急促有爆发力。音调强硬但只在基线附近小幅波动，不要吼到破音变调，"
           "带咆哮和咬牙的感觉，胸腔共鸣重，气息粗重，"
           "句尾像是从喉咙里吼出来的。"),
    "ai": ("哀",
           "怎么会这样……我以为一切都还来得及。唉，现在说什么都晚了。",
           "非常悲伤哽咽的声音。音调比你平时说话明显更低沉、低八度左右，"
           "喉咙发紧像含着眼泪。音量极轻，几乎是气声和耳语，"
           "但每一个字都在明显发抖、声音颤抖不止。"
           "语速非常缓慢，每个字都拖得很长，字与字之间有断续的停顿和抽泣气音，"
           "尾音无力地下垂并带着长长的叹息。"),
}

BASE_LINE = "今天天气不错，我准备去附近走一走，看看有什么新鲜事。"

# 目标区间（f0_ratio / f0std_ratio 均为相对该角色基线的倍率）
TARGETS = {
    "xi": {"rms_db": (-26, -18), "rate": (3.6, 6.0), "f0_ratio": (1.00, 1.30)},
    "nu": {"rms_db": (-26, -17), "rate": (3.2, 5.5), "f0_ratio": (0.95, 1.30),
           "crest": (7.0, 16.0)},
    "ai": {"rms_db": (-30, -22), "rate": (1.4, 2.6), "f0_ratio": (0.82, 1.02),
           "f0std_ratio": (0.08, 0.30)},
}


def read_wav(p):
    w, sr = sf.read(p, dtype="float32")
    if w.ndim > 1:
        w = w[:, 0]
    return w, sr


def f0_track(x, sr):
    hop = int(sr * 0.04)
    win = int(sr * 0.04)
    lo, hi = int(sr / 400), int(sr / 70)
    f0s = []
    for i in range(0, len(x) - win, hop):
        fr = x[i:i + win]
        if np.sqrt(np.mean(fr ** 2)) < 0.005:
            continue
        fr = fr - fr.mean()
        c = np.correlate(fr, fr, "full")[win - 1:]
        if len(c) <= hi:
            continue
        seg = c[lo:hi]
        if len(seg) == 0:
            continue
        k = int(np.argmax(seg)) + lo
        if c[k] <= 0.3 * c[0] or c[k] <= 0:
            continue
        f0s.append(sr / k)
    return np.array(f0s) if f0s else np.array([])


def feats(x, sr, n_syl):
    dur = len(x) / sr
    rms = float(np.sqrt(np.mean(x ** 2)))
    rms_db = 20 * np.log10(max(rms, 1e-9))
    peak = float(np.max(np.abs(x)))
    p95 = float(np.percentile(np.abs(x), 95))
    p95_db = 20 * np.log10(max(p95, 1e-9))
    crest = float(peak / max(rms, 1e-9))
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    f0 = f0_track(x, sr)
    f0_mean = float(np.median(f0)) if len(f0) else 0.0
    f0_std = float(np.std(f0)) if len(f0) else 0.0
    head = x[:int(sr * 0.15)]
    onset_jerk = float(np.max(np.diff(np.abs(head))) / max(np.mean(np.abs(head)), 1e-9)) \
        if len(head) > 16 else 0.0
    hop = int(sr * 0.02)
    fr = x[:len(x) // hop * hop].reshape(-1, hop)
    if len(fr):
        fe = np.sqrt(np.mean(fr ** 2, axis=1))
        thr = 0.2 * (np.percentile(fe, 95) + 1e-9)
        breath = float(np.mean(fe < thr))
    else:
        breath = 0.0
    return {
        "dur": round(dur, 3),
        "rms_db": round(rms_db, 2),
        "p95_db": round(p95_db, 2),
        "crest": round(crest, 3),
        "zcr": round(zcr, 4),
        "f0_mean": round(f0_mean, 1),
        "f0_std": round(f0_std, 1),
        "onset_jerk": round(onset_jerk, 3),
        "breath_ratio": round(breath, 3),
        "rate": round(n_syl / max(dur, 1e-6), 3),
    }


def syl_count(line):
    n = 0
    for ch in line:
        if "一" <= ch <= "鿿":
            n += 1
        elif ch in "！？。":
            n += 0.5
    return max(n, 1)


def score(f, targets, base_f0):
    s = 0.0
    n = 0
    for k, (lo, hi) in targets.items():
        n += 1
        if k == "f0_ratio":
            if base_f0 <= 0:
                s += 0.5
                continue
            v = f["f0_mean"] / base_f0
        elif k == "f0std_ratio":
            if base_f0 <= 0:
                s += 0.5
                continue
            v = f["f0_std"] / base_f0
        else:
            v = f[k]
        if lo <= v <= hi:
            s += 1.0
        else:
            span = (hi - lo) or 1.0
            d = (lo - v) if v < lo else (v - hi)
            s += max(0.0, 1.0 - d / span)
    return max(0.0, s / n) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--cand", type=int, default=3)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--only", default=None, help="逗号分隔 id 白名单（样片用）")
    ap.add_argument("--regen", action="store_true", help="已存在也重新生成")
    ap.add_argument("--emotions", default=None, help="逗号分隔情绪白名单，默认全部")
    args = ap.parse_args()

    raw = json.load(open(ANIMALS, encoding="utf-8"))
    animals = raw["animals"] if isinstance(raw, dict) else raw
    if args.only:
        keep = set(x.strip() for x in args.only.split(',') if x.strip())
        animals = [a for a in animals if a['id'] in keep]
    mine = [(i, a) for i, a in enumerate(animals) if i % args.of == args.shard]
    print(f"[load] shard {args.shard}/{args.of}: {len(mine)} animals x 3 emotions "
          f"x {args.cand} candidates", flush=True)

    # 基线音高按分片存文件，避免多进程并发写同一个 json 互相覆盖
    os.makedirs(BASE_F0_DIR, exist_ok=True)
    base_map = {}
    for bf in sorted(os.listdir(BASE_F0_DIR)):
        if bf.endswith(".json"):
            try:
                base_map.update(json.load(open(os.path.join(BASE_F0_DIR, bf), encoding="utf-8")))
            except Exception:
                pass
    legacy = "/ssd2/duanju/voices/base_f0_animals.json"
    if os.path.exists(legacy):
        try:
            base_map.update(json.load(open(legacy, encoding="utf-8")))
        except Exception:
            pass

    model = Qwen3TTSModel.from_pretrained(
        DESIGN, device_map=f"cuda:{args.gpu}", dtype=torch.bfloat16,
        attn_implementation="sdpa")

    report = []
    for gi, a in mine:
        cid, name = a["id"], a["name"]
        base_desc = a.get("voice_desc", "")
        base_core = base_desc.split("，")[0] if base_desc else f"{name}的声音"
        cdir = os.path.join(OUT, cid)
        os.makedirs(cdir, exist_ok=True)

        # ---- 基线音高 ----
        base_f0 = base_map.get(cid, 0.0)
        if base_f0 <= 0:
            try:
                torch.manual_seed(700000 + gi * 131)
                wavs, sr = model.generate_voice_design(BASE_LINE, base_core, "Chinese")
                w = np.asarray(wavs[0], dtype=np.float32)
                bf = feats(w, sr, syl_count(BASE_LINE))
                base_f0 = bf["f0_mean"]
                base_map[cid] = base_f0
                print(f"  [base] {cid} {name} f0={base_f0}", flush=True)
            except Exception as ex:
                print(f"  [base] {cid} FAIL: {ex}", flush=True)
                base_f0 = 0.0

        for ek, (ezh, line, edesc) in EMOS.items():
            if args.emotions and ek not in [x.strip() for x in args.emotions.split(',')]:
                continue
            path = os.path.join(cdir, f"{cid}_{ek}.wav")
            if os.path.exists(path) and os.path.getsize(path) > 1024 and not args.regen:
                print(f"  [skip] {cid} {ezh} exists", flush=True)
                continue
            n_syl = syl_count(line)
            tgt = TARGETS[ek]
            best = None
            cands = []
            ek_seed = sum(ord(ch) for ch in ek)
            for ci in range(args.cand):
                seed = 700000 + gi * 131 + ek_seed * 7 + ci * 17
                try:
                    torch.manual_seed(seed)
                    desc = f"{base_core}，用{ezh}的情绪演绎这句话：{edesc}"
                    wavs, sr = model.generate_voice_design(line, desc, "Chinese")
                    w = np.asarray(wavs[0], dtype=np.float32)
                    f = feats(w, sr, n_syl)
                    sc = score(f, tgt, base_f0)
                    cands.append({"seed": seed, "score": round(sc, 1), "feats": f})
                    print(f"  [cand] {cid} {ezh} c{ci} score={sc:.1f} "
                          f"f0={f['f0_mean']}(base{base_f0}) rms={f['rms_db']} "
                          f"rate={f['rate']} jitter={f['f0_std']}", flush=True)
                    if best is None or sc > best[0]:
                        best = (sc, seed, w, sr, f)
                except Exception as ex:
                    print(f"  [cand] {cid} {ezh} c{ci} FAIL: {ex}", flush=True)

            if best is None:
                print(f"[FAIL] {cid} {ezh}: all candidates failed", flush=True)
                report.append({"id": cid, "name": name, "emotion": ezh, "emotion_key": ek,
                               "ok": False, "error": "all candidates failed"})
                continue

            if best[0] < 55:
                print(f"  [warn] {cid} {ezh} best_score={best[0]:.1f} 偏低", flush=True)

            sc, seed, w, sr, f = best
            sf.write(path, w, sr)
            print(f"[ok] {cid} {name} {ezh} score={sc:.1f} seed={seed} "
                  f"dur={f['dur']}s f0={f['f0_mean']} rms={f['rms_db']} "
                  f"rate={f['rate']} jitter={f['f0_std']}", flush=True)
            report.append({"id": cid, "name": name, "emotion": ezh, "emotion_key": ek,
                           "path": path, "ok": True, "score": round(sc, 1), "seed": seed,
                           "base_f0": base_f0, "best_feats": f, "candidates": cands})

    mp = f"/ssd2/duanju/voices/animals_voices_shard{args.shard}.json"
    json.dump({"shard": args.shard, "of": args.of, "count": len(report), "items": report},
              open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ok = sum(1 for r in report if r.get("ok"))
    avg = np.mean([r["score"] for r in report if r.get("ok")]) if ok else 0
    print(f"[done] shard{args.shard}: {ok}/{len(report)} avg_score={avg:.1f} -> {mp}", flush=True)

    # 基线音高按分片写回
    bf_path = os.path.join(BASE_F0_DIR, f"base_shard{args.shard}_of{args.of}.json")
    json.dump(base_map, open(bf_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[base_f0] saved {len(base_map)} -> {bf_path}", flush=True)


if __name__ == "__main__":
    main()
