#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对已生成的 900 条日语音色重新打分（只读音频，不重新生成）。"""
import json
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, '/ssd2/duanju/scripts')
import gen_emotion_voices_v2 as V2
from gen_japanese_voices_ja import (JA, ja_syl_count, TARGETS_JA, score_jing,
                                   score_ja)

OUT = "/ssd2/duanju/voices/japanese_ja"
CAST = "/ssd2/duanju/scripts/japanese_100.json"
BASE_F0_DIR = "/ssd2/duanju/voices/base_f0_japanese_ja"
REPORT = "/ssd2/duanju/voices/japanese_ja_voices_rescore.json"

cast = json.load(open(CAST, encoding="utf-8"))
base_map = {}
for bf in sorted(os.listdir(BASE_F0_DIR)):
    if bf.endswith(".json"):
        try:
            base_map.update(json.load(open(os.path.join(BASE_F0_DIR, bf), encoding="utf-8")))
        except Exception:
            pass

items = []
for c in cast:
    cid = c["id"]
    gender = c.get("gender", "")
    base_f0 = base_map.get(cid, 0.0)
    for ek in JA.keys():
        path = os.path.join(OUT, cid, f"{cid}_{ek}.wav")
        if not os.path.exists(path) or os.path.getsize(path) <= 1024:
            items.append({"id": cid, "emotion_key": ek, "ok": False, "error": "missing"})
            continue
        ezh, line, _ = JA[ek]
        try:
            w, sr = sf.read(path, dtype="float32")
            if w.ndim > 1:
                w = w.mean(axis=1)
            f = V2.feats(w, sr, ja_syl_count(line))
            sc = score_jing(f, base_f0, gender) if ek == "jing" \
                else score_ja(f, TARGETS_JA[ek], base_f0, gender, ek)
            items.append({"id": cid, "name": c.get("name", cid), "emotion": ezh,
                          "emotion_key": ek, "path": path, "ok": True,
                          "score": round(sc, 1), "base_f0": base_f0, "feats": f})
        except Exception as ex:
            items.append({"id": cid, "emotion_key": ek, "ok": False, "error": str(ex)})

json.dump({"count": len(items), "items": items}, open(REPORT, "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

ok = [i for i in items if i.get("ok")]
scores = sorted(i["score"] for i in ok)
print(f"报告 {len(items)} 成功 {len(ok)} avg {sum(scores)/max(len(scores),1):.1f} "
      f"median {scores[len(scores)//2] if scores else 0:.1f} min {scores[0] if scores else 0:.1f}")
print(f"<55: {sum(1 for s in scores if s < 55)}  <70: {sum(1 for s in scores if s < 70)}")
for ek in JA.keys():
    ss = [i["score"] for i in ok if i["emotion_key"] == ek]
    if ss:
        print(f"{ek} n {len(ss)} avg {sum(ss)/len(ss):.1f} <55: {sum(1 for s in ss if s < 55)}")
low = sorted([i for i in ok if i["score"] < 55], key=lambda x: x["score"])
print("低分:", ",".join(f"{i['id']}_{i['emotion_key']}({i['score']})" for i in low))
