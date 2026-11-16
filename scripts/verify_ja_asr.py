#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 faster-whisper-large-v3 对日语音色做 ASR 校验（验证是否真在说日语）。"""
import json
import os
import sys

from faster_whisper import WhisperModel

sys.path.insert(0, '/ssd2/duanju/scripts')
from gen_japanese_voices_ja import JA

OUT = "/ssd2/duanju/voices/japanese_ja"
MODEL = "/ssd2/cache/whisper/models--Systran--faster-whisper-large-v3/snapshots"

snap = None
for root, dirs, files in os.walk(MODEL):
    if "model.bin" in files:
        snap = root
        break
print("model:", snap, flush=True)

model = WhisperModel(snap, device="cuda", compute_type="float16")

ids = sys.argv[1].split(',') if len(sys.argv) > 1 else ["F01", "M03", "M22"]
rows = []
for cid in ids:
    for ek, (ezh, line, _) in JA.items():
        p = os.path.join(OUT, cid, f"{cid}_{ek}.wav")
        if not os.path.exists(p):
            continue
        # 自动语言检测：不预设语言，否则任何音频都会被转写成日语（自欺欺人）
        segs, info = model.transcribe(p, language=None, beam_size=5)
        hyp = "".join(s.text for s in segs).strip()
        rows.append({"id": cid, "emo": ek, "ref": line, "hyp": hyp,
                     "dur": info.duration, "lang": info.language,
                     "lang_prob": info.language_probability})

for r in rows:
    print(f"{r['id']}_{r['emo']}\tlang={r['lang']}({r.get('lang_prob',0):.2f})\t{r['dur']:.1f}s\t{r['hyp']}", flush=True)

json.dump(rows, open("/ssd2/duanju/voices/ja_asr_check.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"[done] {len(rows)} 条", flush=True)
