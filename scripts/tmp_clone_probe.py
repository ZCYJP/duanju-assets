#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 generate_voice_clone 能否让情绪音色继承角色主音色的音高本色。

问题背景
--------
用 generate_voice_design(text, instruct) 纯靠文字描述生成情绪音色时，
Qwen3-TTS 在「喜/怒/惧/惊」这类高唤醒情绪下会自发把音高拉高，
男声基线 117Hz 被拉到 260~308Hz —— 直接进入女声音域，形成性别漂移。
文字指令压不住（prompt 层面无解）。

假设
----
Qwen3-TTS-12Hz-1.7B-Base 支持 generate_voice_clone(ref_audio=..., ref_text=...)，
即以角色主音色 wav 为参考克隆音色，再用 instruct 附加情绪描述。
如果克隆能锁住音高区间（与主音色同一性别音域），
则「同一人 + 不同情绪」就能同时成立。

被测项
------
对同一角色（男声 + 女声各一）：
  A. 主音色原始 f0（基线）
  B. voice_design 纯文字指令  -> 预期漂移
  C. voice_clone + 情绪 instruct -> 期望锁住音高
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/ssd2/ttslib")
from qwen_tts import Qwen3TTSModel  # noqa: E402

BASE_MODEL = "/ssd2/models/Qwen3-TTS-12Hz-1.7B-Base"
VOICES = "/ssd2/duanju/voices"
SAMPLES = os.path.join(VOICES, "samples")
SAMPLES_EXT = os.path.join(VOICES, "samples_ext")
GPU = sys.argv[1] if len(sys.argv) > 1 else "0"

BASE = {
    "xi": ("喜", "太好了！哈哈，这下可让我高兴坏了，真是让人忍不住笑出声来！"),
    "nu": ("怒", "够了！你别再说了，这件事我绝不会让步——滚出去！"),
    "ju": ("惧", "别、别过来……我、我什么都没看见……你别再靠近了……"),
    "ai": ("哀", "怎么会这样……我以为一切都还来得及。唉，现在说什么都晚了。"),
}


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
    return np.array(f0s) if len(f0s) else np.array([])


def med_f0(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    f = f0_track(x, sr)
    return (float(np.median(f)) if len(f) else 0.0), len(x) / sr


def find_sample(cid):
    for d in (SAMPLES, SAMPLES_EXT):
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if fn.startswith(cid + "_") and fn.endswith(".wav"):
                return os.path.join(d, fn)
    return None


# 主音色的参考文本：需要与 ref_audio 内容一致，clone 才稳
REF_TEXT = "我是这个世界的主角，我的故事才刚刚开始。"

cast = json.load(open("/ssd2/duanju/scripts/cast_all_150.json", encoding="utf-8"))
# 取 1 男 1 女 做探针
picks = []
for c in cast:
    if c.get("gender") == "男" and not any(p[1] == "男" for p in picks):
        picks.append((c, "男"))
    if c.get("gender") == "女" and not any(p[1] == "女" for p in picks):
        picks.append((c, "女"))
    if len(picks) == 2:
        break

model = Qwen3TTSModel.from_pretrained(
    BASE_MODEL, device_map=f"cuda:{GPU}", dtype=torch.bfloat16,
    attn_implementation="sdpa")
print(f"[load] Base model ok on cuda:{GPU}", flush=True)

for c, g in picks:
    cid = c["id"]
    sp = find_sample(cid)
    print(f"\n===== {cid} {c['name']} ({g}) =====", flush=True)
    if not sp:
        print("  [skip] 找不到主音色样本")
        continue
    bf0, bdur = med_f0(sp)
    print(f"  [base] 主音色 f0={bf0:.1f}Hz dur={bdur:.2f}s  {os.path.basename(sp)}", flush=True)

    ref_x, ref_sr = sf.read(sp, dtype="float32")
    if ref_x.ndim > 1:
        ref_x = ref_x[:, 0]

    for ek, (ezh, line) in BASE.items():
        # --- B. 纯文字 voice_design（对照组）---
        try:
            torch.manual_seed(1234)
            wavs, sr = model.generate_voice_design(line, f"用{ezh}的情绪演绎这句话", "Chinese")
            w = np.asarray(wavs[0], dtype="float32")
            f = f0_track(w, sr)
            m = float(np.median(f)) if len(f) else 0.0
            r = m / bf0 if bf0 else 0
            print(f"  [design ] {ezh} f0={m:.1f} r={r:.2f} dur={len(w)/sr:.2f}s", flush=True)
        except Exception as ex:
            print(f"  [design ] {ezh} FAIL {type(ex).__name__}: {ex}", flush=True)

        # --- C. voice_clone + 情绪 instruct（实验组）---
        try:
            torch.manual_seed(1234)
            wavs, sr = model.generate_voice_clone(
                line, "Chinese",
                ref_audio=(ref_x, ref_sr),
                ref_text=REF_TEXT,
            )
            w = np.asarray(wavs[0], dtype="float32")
            f = f0_track(w, sr)
            m = float(np.median(f)) if len(f) else 0.0
            r = m / bf0 if bf0 else 0
            print(f"  [clone  ] {ezh} f0={m:.1f} r={r:.2f} dur={len(w)/sr:.2f}s", flush=True)
        except Exception as ex:
            print(f"  [clone  ] {ezh} FAIL {type(ex).__name__}: {ex}", flush=True)

print("\n[done]")
