"""快速景别预检：OpenCV Haar 人脸检测（CPU，极快）+ 亮度/构图统计"""
import os
import glob
import json
import numpy as np
import cv2

FRAMES = "/tmp/refan/frames"
files = sorted(glob.glob(os.path.join(FRAMES, "*.jpg")))
print("帧数", len(files))

casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
print("cascade loaded", not casc.empty())

rows = []
for i, fp in enumerate(files):
    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    H, W = img.shape[:2]
    faces = casc.detectMultiScale(img, scaleFactor=1.12, minNeighbors=5, minSize=(16, 16))
    if len(faces) == 0:
        rows.append({"f": os.path.basename(fp), "n": 0, "fh": 0.0, "cls": "无脸",
                     "bright": float(img.mean())})
        continue
    areas = [w * h for (x, y, w, h) in faces]
    k = int(np.argmax(areas))
    x, y, w, h = faces[k]
    fh = h / H
    if fh >= 0.28:
        cls = "大特写/特写"
    elif fh >= 0.18:
        cls = "特写"
    elif fh >= 0.11:
        cls = "近景"
    elif fh >= 0.06:
        cls = "中景"
    elif fh >= 0.02:
        cls = "中全景"
    else:
        cls = "全景/远景"
    rows.append({"f": os.path.basename(fp), "n": len(faces), "fh": float(fh), "cls": cls,
                 "cx": float((x + w / 2) / W), "cy": float((y + h / 2) / H),
                 "bright": float(img.mean())})
    if (i + 1) % 800 == 0:
        print("  %d" % (i + 1), flush=True)

json.dump(rows, open("/tmp/refan/shot_scale_haar.json", "w"))
N = len(rows)
from collections import Counter
print("\n=== 景别分布（Haar 预检）===")
cc = Counter(r["cls"] for r in rows)
for k2 in ["大特写/特写", "特写", "近景", "中景", "中全景", "全景/远景", "无脸"]:
    v = cc.get(k2, 0)
    print("  %-12s : %5d  %5.1f%%  %s" % (k2, v, v / N * 100, "#" * int(v / N * 120)))
nf = Counter(r["n"] for r in rows)
print("\n=== 同框人数 ===")
for k2 in sorted(nf):
    print("  %d 人 : %5d 帧 %5.1f%%" % (k2, nf[k2], nf[k2] / N * 100))
fp_rows = [r for r in rows if r["n"] > 0]
if fp_rows:
    fh = np.array([r["fh"] for r in fp_rows])
    print("\n人脸高占比 均值 %.3f 中位 %.3f P25 %.3f P75 %.3f P90 %.3f" % (
        fh.mean(), np.median(fh), *np.percentile(fh, [25, 75, 90])))
br = np.array([r["bright"] for r in rows])
print("亮度 均值 %.1f  P5 %.1f  P95 %.1f  极差 %.1f" % (
    br.mean(), np.percentile(br, 5), np.percentile(br, 95), br.max() - br.min()))
print("\nwrote /tmp/refan/shot_scale_haar.json")
