#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""100 个日本角色 × 9 种状态音色（喜/怒/哀/乐/惧/惊/厌/忧/羞）。

复用 gen_emotion_voices_v2.py 的情绪描述、目标区间与打分函数（人物版），
新增：先为每个角色生成一条中性基线音频测出 f0_mean 作为基线，
      音高与抖动一律按「相对基线的倍率」判定，避免 VoiceDesign 随机音高导致漂移。

用法：
  python3 gen_japanese_voices.py --shard 0 --of 4 --gpu 1 --cand 3
输出：/ssd2/duanju/voices/japanese/<ID>/<ID>_<emo>.wav
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

sys.path.insert(0, '/ssd2/duanju/scripts')
import gen_emotion_voices_v2 as V2  # 复用 BASE / DESC_V2 / DESC_V3 / TARGETS / feats / score

OUT = "/ssd2/duanju/voices/japanese"
DESIGN = os.environ.get("QWEN_TTS_DESIGN", "/ssd2/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
CAST = os.environ.get("CAST_JSON", "/ssd2/duanju/scripts/japanese_100.json")
BASE_F0_DIR = "/ssd2/duanju/voices/base_f0_japanese"

BASE_LINE = "今天天气不错，我准备去附近走一走，看看有什么新鲜事。"

ALL_DESC = dict(V2.DESC_V2)
ALL_DESC.update(V2.DESC_V3)

# 「惊」的打分口径修正（与 rescore_japanese_voices.py 保持一致）：
# 原 V2.score 对 jing 用「相对基线倍率 1.05~1.40」+「性别相对倍率硬拦截」，
# 而 VoiceDesign 的惊叫天然把音高拉到基线的 2.4 倍（低音高男声 107→258Hz），
# 于是所有候选 score=0，选优退化为「取第一个候选」→ 惊情绪整批低分。
# 修正：倍率放宽到 2.60，性别漂移改用绝对音高边界（男 70~320Hz / 女 110~430Hz）。
JING_F0_RATIO = (1.05, 2.60)
JING_TARGET = {"onset_jerk": (1.5, 12.0), "f0_mean": (240, 400), "rate": (2.6, 6.0)}
GENDER_ABS_F0 = {"男": (70.0, 320.0), "女": (110.0, 430.0)}


def score_jing(f, base_f0, gender):
    s = 0.0
    n = 0
    for k, (lo, hi) in JING_TARGET.items():
        v = f[k]
        w = 3.0 if k == "f0_mean" else 1.0
        n += w
        if k == "f0_mean":
            if base_f0 > 0:
                r = v / base_f0
                if JING_F0_RATIO[0] <= r <= JING_F0_RATIO[1]:
                    s += w
                else:
                    span = (JING_F0_RATIO[1] - JING_F0_RATIO[0]) or 1.0
                    d = (JING_F0_RATIO[0] - r) if r < JING_F0_RATIO[0] else (r - JING_F0_RATIO[1])
                    s += w * max(0.0, 1.0 - d / span)
            else:
                s += w * 0.5
            glo, ghi = GENDER_ABS_F0.get(gender, (60.0, 450.0))
            if v < glo or v > ghi:
                s -= w
            continue
        if lo <= v <= hi:
            s += w
        else:
            span = (hi - lo) or 1.0
            d = (lo - v) if v < lo else (v - hi)
            s += w * max(0.0, 1.0 - d / span)
    return max(0.0, s / n) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--cand", type=int, default=3)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--only", default=None, help="逗号分隔 id 白名单（样片用）")
    ap.add_argument("--emotions", default=None, help="逗号分隔情绪白名单，默认全部 9 种")
    ap.add_argument("--regen", action="store_true", help="已存在也重新生成")
    args = ap.parse_args()

    cast = json.load(open(CAST, encoding="utf-8"))
    if args.only:
        keep = set(x.strip() for x in args.only.split(',') if x.strip())
        cast = [c for c in cast if c['id'] in keep]
    mine = [(i, c) for i, c in enumerate(cast) if i % args.of == args.shard]
    emos = [e.strip() for e in (args.emotions or ",".join(V2.BASE.keys())).split(',') if e.strip()]
    emos = [e for e in emos if e in ALL_DESC]
    print(f"[load] shard {args.shard}/{args.of}: {len(mine)} roles x {len(emos)} emotions "
          f"x {args.cand} candidates", flush=True)

    os.makedirs(BASE_F0_DIR, exist_ok=True)
    base_map = {}
    for bf in sorted(os.listdir(BASE_F0_DIR)):
        if bf.endswith(".json"):
            try:
                base_map.update(json.load(open(os.path.join(BASE_F0_DIR, bf), encoding="utf-8")))
            except Exception:
                pass

    model = Qwen3TTSModel.from_pretrained(
        DESIGN, device_map=f"cuda:{args.gpu}", dtype=torch.bfloat16,
        attn_implementation="sdpa")

    report = []
    for gi, c in mine:
        cid, name = c["id"], c.get("name", c["id"])
        base_desc = c.get("voice_desc", "")
        base_core = base_desc.split("，")[0] if base_desc else ""
        gender = c.get("gender", "")
        cdir = os.path.join(OUT, cid)
        os.makedirs(cdir, exist_ok=True)

        # ---- 基线音高 ----
        base_f0 = base_map.get(cid, 0.0)
        if base_f0 <= 0:
            try:
                torch.manual_seed(930000 + gi * 131)
                wavs, sr = model.generate_voice_design(BASE_LINE, base_core, "Chinese")
                w = np.asarray(wavs[0], dtype=np.float32)
                bf = V2.feats(w, sr, V2.syl_count(BASE_LINE))
                base_f0 = bf["f0_mean"]
                base_map[cid] = base_f0
                print(f"  [base] {cid} {name} f0={base_f0}", flush=True)
            except Exception as ex:
                print(f"  [base] {cid} FAIL: {ex}", flush=True)
                base_f0 = 0.0

        for ek in emos:
            _, ezh, line = V2.BASE[ek]
            edesc = ALL_DESC[ek][1]
            path = os.path.join(cdir, f"{cid}_{ek}.wav")
            if os.path.exists(path) and os.path.getsize(path) > 1024 and not args.regen:
                print(f"  [skip] {cid} {ezh} exists", flush=True)
                continue
            n_syl = V2.syl_count(line)
            tgt = V2.TARGETS[ek]
            best = None
            cands = []
            ek_seed = sum(ord(ch) for ch in ek)
            for ci in range(args.cand):
                seed = 930000 + gi * 131 + ek_seed * 7 + ci * 17
                try:
                    torch.manual_seed(seed)
                    desc = f"{base_core}，用{ezh}的情绪演绎这句话：{edesc}"
                    wavs, sr = model.generate_voice_design(line, desc, "Chinese")
                    w = np.asarray(wavs[0], dtype=np.float32)
                    f = V2.feats(w, sr, n_syl)
                    sc = score_jing(f, base_f0, gender) if ek == "jing" \
                        else V2.score(f, tgt, base_f0, gender, ek)
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

    mp = f"/ssd2/duanju/voices/japanese_voices_shard{args.shard}.json"
    json.dump({"shard": args.shard, "of": args.of, "emotions": emos,
               "count": len(report), "items": report},
              open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ok = sum(1 for r in report if r.get("ok"))
    avg = np.mean([r["score"] for r in report if r.get("ok")]) if ok else 0
    print(f"[done] shard{args.shard}: {ok}/{len(report)} avg_score={avg:.1f} -> {mp}", flush=True)

    bf_path = os.path.join(BASE_F0_DIR, f"base_shard{args.shard}_of{args.of}.json")
    json.dump(base_map, open(bf_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[base_f0] saved {len(base_map)} -> {bf_path}", flush=True)


if __name__ == "__main__":
    main()
