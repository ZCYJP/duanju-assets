#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复音色情调漂移：对 F0 超出性别合理区间的角色，换 seed 重新生成。

男声合理区间：90-215Hz（>220Hz 视为漂移）
女声合理区间：155-300Hz（<150Hz 视为漂移；150-175Hz 仅标记 LOW-F，
             多为老年女性角色，属正常人设，不做重生成）

用法：python3 tmp_fix_voices.py --ids M36,M27,M73,M78,M69
"""
import argparse, json, os, sys
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

OUT = "/ssd2/duanju/voices/samples_ext"
DESIGN = os.environ.get("QWEN_TTS_DESIGN", "/ssd2/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
CAST = "/ssd2/duanju/scripts/cast_all_150.json"
SAMPLE = "这条路我走了很久，今天，终于轮到我来定规矩。"


def rough_f0(x, sr):
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
    ap.add_argument('--ids', required=True)
    ap.add_argument('--tries', type=int, default=8)
    args = ap.parse_args()

    ids = [x.strip() for x in args.ids.split(',') if x.strip()]
    cast = {c['id']: c for c in json.load(open(CAST, encoding='utf-8'))}

    model = Qwen3TTSModel.from_pretrained(
        DESIGN, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")

    fixed, failed = [], []
    for cid in ids:
        c = cast[cid]
        gender = "female" if c['gender'] == "女" else "male"
        desc = c['voice_desc']
        path = f"{OUT}/{cid}_{c['name']}_voice.wav"
        base = 8000 + int(cid[1:]) * 17

        # 读当前 F0 作为基线
        w0, sr0 = sf.read(path, dtype="float32")
        if w0.ndim > 1:
            w0 = w0[:, 0]
        f0_cur = rough_f0(w0, sr0)
        print(f"[{cid} {c['name']}] current F0={f0_cur:.0f}Hz", flush=True)

        lo, hi = (90, 215) if gender == "male" else (155, 300)
        ok = lo <= f0_cur <= hi
        if ok:
            print(f"  already OK, skip", flush=True)
            fixed.append({"id": cid, "f0": round(f0_cur, 1), "changed": False})
            continue

        best = None
        for k in range(1, args.tries + 1):
            seed = base + 1000 + k * 37
            torch.manual_seed(seed)
            wavs, sr = model.generate_voice_design(SAMPLE, desc, "Chinese")
            w = np.asarray(wavs[0], dtype=np.float32)
            f0 = rough_f0(w, sr)
            print(f"  try seed={seed} -> F0={f0:.0f}Hz", flush=True)
            if lo <= f0 <= hi:
                sf.write(path, w, sr)
                print(f"  FIXED {cid} seed={seed} F0={f0:.0f}Hz", flush=True)
                fixed.append({"id": cid, "f0": round(f0, 1), "seed": seed, "changed": True})
                best = f0
                break
            if best is None or abs(f0 - 190) < abs(best - 190):
                best, best_w, best_sr = f0, w, sr
        if best is None or not (lo <= best <= hi):
            # 保留最接近中间值的一版
            sf.write(path, best_w, best_sr)
            print(f"  BEST-EFFORT {cid} F0={best:.0f}Hz (range {lo}-{hi})", flush=True)
            failed.append({"id": cid, "f0": round(best, 1), "range": [lo, hi]})

    json.dump({"fixed": fixed, "failed": failed},
              open("/ssd2/duanju/voices/fix_report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[done] fixed={len(fixed)} best_effort={len(failed)}", flush=True)


if __name__ == "__main__":
    main()
