#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""合并全部音色 manifest（50 人 + 100 人）为统一的 voice_manifest_all.json，
并对每个 wav 做时长/F0/峰值校验，输出总表。

输出：/ssd2/duanju/voices/voice_manifest_all.json
"""
import glob, json, os
import numpy as np
import soundfile as sf

VOICES = "/ssd2/duanju/voices"
CAST = "/ssd2/duanju/scripts/cast_all_150.json"
OUT = os.path.join(VOICES, "voice_manifest_all.json")


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
    cast = {c['id']: c for c in json.load(open(CAST, encoding='utf-8'))}
    items = []
    for p in sorted(glob.glob(os.path.join(VOICES, "samples", "*.wav")) +
                    glob.glob(os.path.join(VOICES, "samples_ext", "*.wav"))):
        base = os.path.basename(p)[:-4]
        cid = base.split('_')[0]
        c = cast.get(cid)
        if not c:
            print(f"  WARN unknown id {cid} in {p}")
            continue
        try:
            w, sr = sf.read(p, dtype="float32")
            if w.ndim > 1:
                w = w[:, 0]
            f0 = rough_f0(w, sr)
            dur = len(w) / sr
            peak = float(np.max(np.abs(w))) if len(w) else 0.0
            gender = "female" if c['gender'] == "女" else "male"
            drift = (gender == "female" and f0 and f0 < 150) or (gender == "male" and f0 and f0 > 220)
            items.append({
                "id": cid, "name": c['name'], "gender": gender, "age": c['age'],
                "role": c.get("role", ""), "voice_desc": c.get("voice_desc", ""),
                "path": p, "dur": round(dur, 2), "f0": round(f0, 1),
                "peak": round(peak, 3), "drift": bool(drift), "sr": sr, "ok": True,
            })
        except Exception as ex:
            items.append({"id": cid, "name": c.get("name", ""), "path": p,
                          "ok": False, "error": str(ex)})

    # F0 唯一性检查：不同角色的音色不应完全一样
    from collections import Counter
    f0s = [it["f0"] for it in items if it.get("ok")]
    dup_f0 = {k: v for k, v in Counter(f0s).items() if v > 1}
    ids = [it["id"] for it in items]
    json.dump({"host": "osa00-h100-038", "count": len(items),
               "ok": sum(1 for it in items if it.get("ok")),
               "drift": sum(1 for it in items if it.get("drift")),
               "unique_ids": len(set(ids)),
               "dup_f0_groups": dup_f0,
               "items": items}, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[done] {len(items)} voices, ok={sum(1 for it in items if it.get('ok'))}, "
          f"drift={sum(1 for it in items if it.get('drift'))}, "
          f"unique_ids={len(set(ids))}/{len(ids)} -> {OUT}")
    print(f"  F0 完全相同的组: {len(dup_f0)}")


if __name__ == "__main__":
    main()
