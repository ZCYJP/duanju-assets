#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为新增 100 个角色各设计一个独特音色（Qwen3-TTS VoiceDesign）。

与 gen_voices_50.py 的差异：
  * 只处理 cast_all_150.json 中编号 >= 26 的新增角色
  * 输出到 /ssd2/duanju/voices/samples_ext（与前 50 人分目录）
  * seed 基于角色 ID 全局唯一（8000 + 全局序号*17），与前 50 人的 7000 段不冲突
  * 多卡并行：--shard/--of 分片，每进程独占一张卡的剩余显存

输出：/ssd2/duanju/voices/samples_ext/<ID>_<name>_voice.wav
      /ssd2/duanju/voices/manifest_ext_shard<i>.json
"""
import argparse, json, os, sys
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

OUT = "/ssd2/duanju/voices/samples_ext"
DESIGN = os.environ.get("QWEN_TTS_DESIGN", "/ssd2/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
CAST = os.environ.get("CAST_JSON", "/ssd2/duanju/scripts/cast_all_150.json")
os.makedirs(OUT, exist_ok=True)

SAMPLE = "这条路我走了很久，今天，终于轮到我来定规矩。"


def rough_f0(x, sr):
    """自相关粗估基频中位数（Hz），只用于发现明显的性别漂移。"""
    frame = int(sr * 0.04)
    hop = int(sr * 0.02)
    vals = []
    for st in range(0, len(x) - frame, hop):
        seg = x[st:st + frame].astype(np.float64)
        if np.sqrt(np.mean(seg ** 2)) < 0.012:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, mode="full")[frame - 1:]
        lo, hi = int(sr / 400), int(sr / 70)
        if hi >= len(ac):
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        vals.append(sr / lag)
    return float(np.median(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--of', type=int, default=1)
    args = ap.parse_args()

    cast = json.load(open(CAST, encoding="utf-8"))
    # 只处理新增角色：编号 >= 26
    cast = [c for c in cast if c.get("voice_desc") and int(c["id"][1:]) >= 26]
    mine = [(i, c) for i, c in enumerate(cast) if i % args.of == args.shard]
    print(f"[load] shard {args.shard}/{args.of}: {len(mine)} characters", flush=True)

    print(f"[load] Qwen3-TTS VoiceDesign <- {DESIGN}", flush=True)
    model = Qwen3TTSModel.from_pretrained(
        DESIGN, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")

    report = []
    for gi, c in mine:
        cid, name = c["id"], c["name"]
        gender = "female" if c["gender"] == "女" else "male"
        desc = c["voice_desc"]
        # seed 与前 50 人（7000 段）错开，且全局唯一
        seed = 8000 + gi * 17
        torch.manual_seed(seed)
        path = f"{OUT}/{cid}_{name}_voice.wav"
        if os.path.exists(path) and os.path.getsize(path) > 10000:
            try:
                w0, sr0 = sf.read(path, dtype="float32")
                if w0.ndim > 1:
                    w0 = w0[:, 0]
                f0 = rough_f0(w0, sr0)
                dur = len(w0) / sr0
                peak = float(np.max(np.abs(w0))) if len(w0) else 0.0
                drift = (gender == "female" and f0 and f0 < 150) or (gender == "male" and f0 and f0 > 220)
                report.append({"id": cid, "name": name, "gender": gender, "role": c.get("role", ""),
                               "desc": desc, "seed": seed, "dur": round(dur, 2), "f0": round(f0, 1),
                               "peak": round(peak, 3), "drift": bool(drift),
                               "low_female": bool(gender == "female" and 150 <= f0 <= 175),
                               "sr": sr0, "ok": True, "path": path, "cached": True})
                print(f"[skip] {cid} {name} exists ({dur:.1f}s, F0≈{f0:.0f}Hz)", flush=True)
            except Exception as ex:
                report.append({"id": cid, "name": name, "gender": gender, "desc": desc,
                               "seed": seed, "ok": False, "error": f"cache_read: {ex}"})
            continue
        try:
            wavs, sr = model.generate_voice_design(SAMPLE, desc, "Chinese")
        except Exception as ex:
            print(f"[FAIL] {cid} {name}: {ex}", flush=True)
            report.append({"id": cid, "name": name, "gender": gender, "desc": desc,
                           "seed": seed, "ok": False, "error": str(ex)})
            continue
        w = np.asarray(wavs[0], dtype=np.float32)
        sf.write(path, w, sr)
        f0 = rough_f0(w, sr)
        dur = len(w) / sr
        peak = float(np.max(np.abs(w)))
        drift = (gender == "female" and f0 and f0 < 150) or (gender == "male" and f0 and f0 > 220)
        low_female = gender == "female" and 150 <= f0 <= 175
        flag = "DRIFT" if drift else ("LOW-F" if low_female else "OK")
        print(f"[ok] {cid} {name}  {dur:.1f}s  F0≈{f0:.0f}Hz  peak={peak:.2f}  {flag}", flush=True)
        report.append({"id": cid, "name": name, "gender": gender, "role": c.get("role", ""),
                       "desc": desc, "seed": seed, "dur": round(dur, 2), "f0": round(f0, 1),
                       "peak": round(peak, 3), "drift": bool(drift),
                       "low_female": bool(low_female), "sr": sr, "ok": True, "path": path})

    mp = f"/ssd2/duanju/voices/manifest_ext_shard{args.shard}.json"
    json.dump({"host": "osa00-h100-038", "model": DESIGN, "sample": SAMPLE,
               "shard": args.shard, "of": args.of, "count": len(report), "items": report},
              open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ok = sum(1 for r in report if r.get("ok"))
    drift = sum(1 for r in report if r.get("drift"))
    print(f"[done] shard{args.shard}: {ok}/{len(report)} voices, {drift} drift -> {mp}", flush=True)


if __name__ == "__main__":
    main()
