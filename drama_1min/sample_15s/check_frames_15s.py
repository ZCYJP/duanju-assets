"""底图景别门禁：SCRFD 测人脸高占比 + 同框人数，对照 shots_15s.py 的 face_ratio 目标"""
import os
import sys
import glob
import json
import numpy as np
import cv2

sys.path.insert(0, "/root/.insightface")
from insightface.app import FaceAnalysis

sys.path.insert(0, "/ssd2/duanju/drama_1min/sample_15s")
from shots_15s import SHOTS

FRAMES = "/ssd2/duanju/drama_1min/sample_15s/frames"

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
det = app.det_model

rows = []
for shot in SHOTS:
    fp = f"{FRAMES}/{shot['name']}.png"
    if not os.path.exists(fp):
        print(f"  [MISSING] {shot['name']}")
        continue
    img = cv2.imread(fp)
    H, W = img.shape[:2]
    try:
        bbs, _ = det.detect(img, max_num=20, metric="default")
    except Exception as e:
        print("  detect err", e)
        bbs = None
    if bbs is None or len(bbs) == 0:
        rows.append({"name": shot["name"], "n": 0, "fh": 0.0, "target": shot["face_ratio"],
                     "framing": shot["framing"], "ok": False})
        continue
    areas = (bbs[:, 2] - bbs[:, 0]) * (bbs[:, 3] - bbs[:, 1])
    order = np.argsort(-areas)
    fh = (bbs[order[0], 3] - bbs[order[0], 1]) / H
    n = len(bbs)
    want_n = 2 if shot["framing"] == "two_shot" else 1
    fh_ok = abs(fh - shot["face_ratio"]) <= 0.09
    n_ok = n >= want_n
    rows.append({"name": shot["name"], "n": int(n), "fh": float(fh),
                 "target": shot["face_ratio"], "framing": shot["framing"],
                 "ok": bool(fh_ok and n_ok), "fh_ok": bool(fh_ok), "n_ok": bool(n_ok),
                 "cx_main": float(((bbs[order[0], 0] + bbs[order[0], 2]) / 2) / W)})

print("\n=== 底图景别门禁 ===")
print("%-24s %-10s %6s %6s %6s  %s" % ("shot", "framing", "目标", "实测", "人数", "判定"))
nfail = 0
for r in rows:
    if r["n"] == 0:
        print("%-24s %-10s %6.2f %6s %6d  FAIL(无人脸)" % (
            r["name"], r["framing"], r["target"], "-", 0))
        nfail += 1
        continue
    flag = "OK" if r["ok"] else ("FAIL" if not (r["fh_ok"] and r["n_ok"]) else "?")
    if flag == "FAIL":
        nfail += 1
    print("%-24s %-10s %6.2f %6.3f %6d  %s  %s" % (
        r["name"], r["framing"], r["target"], r["fh"], r["n"], flag,
        "" if r["fh_ok"] else "景别偏差大"))
print(f"\n通过 {len(rows)-nfail}/{len(rows)}")
json.dump(rows, open("/ssd2/duanju/drama_1min/sample_15s/logs/frame_gate.json", "w"),
          ensure_ascii=False, indent=2)
