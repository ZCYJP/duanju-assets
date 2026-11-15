#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 50 人数字人角色库各设计一个独特音色（Qwen3-TTS VoiceDesign）。

每个角色的音色描述来自 cast.json 的 voice_desc 字段，
每个人生成 1 条样音，另做基频(F0)粗检防止性别漂移。

多卡并行：每张卡跑一个进程，用 --shard/--of 分片不同角色。
每个进程写自己的 manifest 分片，最后由 merge 脚本汇总。

输出：/ssd2/duanju/voices/samples/<ID>_<name>_voice.wav
      /ssd2/duanju/voices/manifest_shard<i>.json
"""
import argparse, json, os, sys
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

OUT = "/ssd2/duanju/voices/samples"
DESIGN = os.environ.get("QWEN_TTS_DESIGN", "/ssd2/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
CAST = os.environ.get("CAST_JSON", "/ssd2/duanju/characters/cast.json")
os.makedirs(OUT, exist_ok=True)

# 每个角色一句试听台词（体现身份气质），带角色名做区分
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
    ap.add_argument('--start', type=int, default=0,
                    help='全局起始序号（续跑用，避免与已完成分片重复）')
    args = ap.parse_args()

    cast = json.load(open(CAST, encoding="utf-8"))
    # 只处理带 voice_desc 的角色
    cast = [c for c in cast if c.get("voice_desc")]
    # 按全局序号分片，保证 seed 在所有分片间全局唯一
    mine = [(i, c) for i, c in enumerate(cast)
            if i % args.of == args.shard and i >= args.start]
    print(f"[load] shard {args.shard}/{args.of}: {len(mine)} characters", flush=True)

    print(f"[load] Qwen3-TTS VoiceDesign <- {DESIGN}", flush=True)
    model = Qwen3TTSModel.from_pretrained(
        DESIGN, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")

    report = []
    for gi, c in mine:
        cid, name = c["id"], c["name"]
        gender = "female" if c["gender"] == "女" else "male"
        desc = c["voice_desc"]
        seed = 7000 + gi * 17
        torch.manual_seed(seed)
        path = f"{OUT}/{cid}_{name}_voice.wav"
        if os.path.exists(path) and os.path.getsize(path) > 10000:
            # 已存在：跳过生成，但仍补进报告，保证 manifest 覆盖全部角色
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

    mp = f"/ssd2/duanju/voices/manifest_shard{args.shard}.json"
    json.dump({"host": "osa00-h100-038", "model": DESIGN, "sample": SAMPLE,
               "shard": args.shard, "of": args.of, "count": len(report), "items": report},
              open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ok = sum(1 for r in report if r.get("ok"))
    drift = sum(1 for r in report if r.get("drift"))
    print(f"[done] shard{args.shard}: {ok}/{len(report)} voices, {drift} drift -> {mp}", flush=True)


if __name__ == "__main__":
    main()
