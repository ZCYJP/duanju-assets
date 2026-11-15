#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重生成 F0 漂移的角色音色（M21 洛星辰），尝试多个 seed 直到基频落入合理区间。"""
import json, sys
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/ssd2/duanju/scripts")
from qwen_tts import Qwen3TTSModel
from gen_voices_50 import rough_f0, SAMPLE

CAST = "/ssd2/duanju/characters/cast.json"
DESIGN = "/ssd2/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
OUT = "/ssd2/duanju/voices/samples"

cast = json.load(open(CAST, encoding="utf-8"))
targets = {t: (lo, hi) for t, lo, hi in
           [(a.split(":")[0], int(a.split(":")[1]), int(a.split(":")[2])) for a in sys.argv[1:]]}

model = Qwen3TTSModel.from_pretrained(DESIGN, device_map="cuda:0",
                                      dtype=torch.bfloat16, attn_implementation="sdpa")

for cid, (lo, hi) in targets.items():
    c = [x for x in cast if x["id"] == cid][0]
    path = f"{OUT}/{cid}_{c['name']}_voice.wav"
    best = None
    for seed in (9111, 9222, 9333, 9444, 9555):
        torch.manual_seed(seed)
        wavs, sr = model.generate_voice_design(SAMPLE, c["voice_desc"], "Chinese")
        w = np.asarray(wavs[0], dtype=np.float32)
        f0 = rough_f0(w, sr)
        ok = lo <= f0 <= hi
        print(f"  {cid} seed={seed} F0={f0:.0f}Hz dur={len(w)/sr:.1f}s "
              f"-> {'KEEP' if ok else 'retry'}", flush=True)
        if ok:
            sf.write(path, w, sr)
            best = {"id": cid, "seed": seed, "f0": round(f0, 1), "path": path}
            break
        if best is None or abs(f0 - (lo + hi) / 2) < abs(best["f0"] - (lo + hi) / 2):
            best = {"id": cid, "seed": seed, "f0": round(f0, 1), "path": path,
                    "w": w, "sr": sr}
    if best and best.get("seed") not in (9111, 9222, 9333, 9444, 9555):
        pass
    if best and "w" in best:
        # 没有完全落入区间，用最接近的
        sf.write(path, best["w"], best["sr"])
        print(f"  {cid} fallback seed={best['seed']} F0={best['f0']}Hz", flush=True)
    elif best:
        print(f"  {cid} fixed seed={best['seed']} F0={best['f0']}Hz", flush=True)
