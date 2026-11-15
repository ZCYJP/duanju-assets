#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重跑单句配音并挑选通过 F0 门禁的 seed。
只重生成指定镜头的配音，qc 用与 qc_voice.py 相同的逐帧 RMS 门限 + 自相关算法。

用法：fix_s02_voice.py s02_ally_lawyer 苏蔓
"""
import os, sys, json, wave, traceback
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from math import gcd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shots_1min import SHOTS, NAME2CHAR, VOICE_REF

BASE = "/ssd2/duanju/drama_1min"
VOICE_DIR = f"{BASE}/voice"
RAW_DIR = f"{VOICE_DIR}/raw"
MODEL_DIR = "/ssd2/models/Qwen3-TTS-12Hz-1.7B-Base"
REF_TEXT = "一人之身，一國之象。把身体当成一个国家来看：精神是君主，气是百姓，爱气才能全身。"
SR, CH, SW = 44100, 1, 2
SEEDS = [42, 43, 44, 45, 46, 47, 48]
FEMALE = {"F01", "F03", "F08"}
MALE = {"M01", "M03"}


def rough_f0(x, sr):
    frame, hop = int(sr * 0.04), int(sr * 0.02)
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
        vals.append(sr / (lo + int(np.argmax(ac[lo:hi]))))
    return float(np.median(vals)) if vals else 0.0


def qc(w, sr, gender, text):
    issues = []
    dur = len(w) / sr
    peak = float(np.max(np.abs(w))) if len(w) else 0.0
    f0 = rough_f0(w, sr)
    cps = len(text) / max(dur, 0.01)
    if dur < 0.4:
        issues.append(f"过短{dur:.2f}s")
    if dur > 9.0:
        issues.append(f"过长{dur:.2f}s")
    if peak > 0.98:
        issues.append(f"削波peak={peak:.3f}")
    if peak < 0.05:
        issues.append(f"音量过低peak={peak:.3f}")
    if not (2.5 <= cps <= 9.0):
        issues.append(f"语速异常{cps:.1f}字/秒")
    if f0 <= 0:
        issues.append("F0无法测量")
    elif gender == "female":
        if f0 < 150:
            issues.append(f"女声F0={f0:.0f}Hz<150 熔断")
        elif f0 < 175:
            issues.append(f"女声F0={f0:.0f}Hz<175")
    else:
        if not (70 <= f0 <= 220):
            issues.append(f"男声F0={f0:.0f}Hz越界")
    return {"issues": issues, "dur": round(dur, 3), "peak": round(peak, 4),
            "f0": round(f0, 1), "cps": round(cps, 2), "gender": gender}


def to_pcm(w, sr, gain=0.92):
    if sr != SR:
        g = gcd(SR, sr)
        w = resample_poly(w, SR // g, sr // g).astype(np.float32)
    w = np.clip(w * gain, -0.99, 0.99)
    return (w * 32767).astype(np.int16).tobytes()


def write_wav(path, pcm_bytes):
    with wave.open(path, "wb") as w:
        w.setnchannels(CH)
        w.setsampwidth(SW)
        w.setframerate(SR)
        w.writeframes(pcm_bytes)


def main():
    shot = sys.argv[1]
    speaker = sys.argv[2]
    cid = NAME2CHAR[speaker]
    gender = "female" if cid in FEMALE else "male"
    text = None
    for s in SHOTS:
        if s["name"] == shot:
            for sp, tx in s["dialogue"]:
                if sp == speaker:
                    text = tx
    if text is None:
        print("找不到台词:", shot, speaker)
        return 1

    rp = VOICE_REF[cid][0]
    print(f"[fix] {shot}_{speaker} cid={cid} gender={gender} text={text}")
    print(f"[fix] ref={rp}")

    from qwen_tts import Qwen3TTSModel
    gpu = os.environ.get("TTS_GPU", "1")
    model = Qwen3TTSModel.from_pretrained(
        MODEL_DIR, device_map=f"cuda:{gpu}", dtype=torch.bfloat16, attn_implementation="sdpa")
    profile = model.create_voice_clone_prompt(rp, REF_TEXT)[0]
    print("[fix] 声纹 profile OK")

    raw_path = f"{RAW_DIR}/{shot}_{speaker}.wav"
    out_path = f"{VOICE_DIR}/{shot}_{speaker}.wav"
    best = None
    for seed in SEEDS:
        torch.manual_seed(seed)
        wavs, sr = model.generate_voice_clone(text, "Chinese", voice_clone_prompt=[profile])
        w = np.asarray(wavs[0], dtype=np.float32)
        r = qc(w, sr, gender, text)
        r["seed"] = seed
        sf.write(f"{RAW_DIR}/{shot}_{speaker}_s{seed}.wav", w, sr)
        if not r["issues"]:
            write_wav(out_path, to_pcm(w, sr))
            print(f"[OK ] seed={seed} dur={r['dur']}s f0={r['f0']}Hz cps={r['cps']} "
                  f"peak={r['peak']} -> 已写入 {out_path}")
            return 0
        print(f"[retry] seed={seed} f0={r['f0']}Hz {r['issues']}")
        if best is None or len(r["issues"]) < len(best[2]["issues"]):
            best = (w, sr, r)

    if best:
        w, sr, r = best
        write_wav(out_path, to_pcm(w, sr))
        print(f"[BEST] seed={best[2]['seed']} f0={r['f0']}Hz {r['issues']} -> 仍写入 {out_path}")
        return 2
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
