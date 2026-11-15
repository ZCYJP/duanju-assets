"""参考片景别分析：用 insightface 检测人脸框占比 → 分类景别；同时统计人物数、画面结构"""
import os
import glob
import json
import numpy as np
import cv2
from insightface.app import FaceAnalysis

FRAMES = "/tmp/refan/frames"
OUT = "/tmp/refan/shot_scale.json"

app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

files = sorted(glob.glob(os.path.join(FRAMES, "*.jpg")))
print("帧数", len(files))

rows = []
for idx, fp in enumerate(files):
    img = cv2.imread(fp)
    if img is None:
        continue
    H, W = img.shape[:2]
    try:
        faces = app.get(img)
    except Exception:
        faces = []
    n = len(faces)
    if n == 0:
        rows.append({"f": os.path.basename(fp), "n": 0, "max_face_h": 0.0,
                     "max_face_w": 0.0, "cls": "无脸"})
        continue
    # 取最大人脸
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    x1, y1, x2, y2 = best.bbox
    fh = (y2 - y1) / H
    fw = (x2 - x1) / W
    # 景别分类（按人脸高度占画面比例）
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
    rows.append({"f": os.path.basename(fp), "n": n, "max_face_h": float(fh),
                 "max_face_w": float(fw), "cls": cls,
                 "cx": float(((x1 + x2) / 2) / W), "cy": float(((y1 + y2) / 2) / H)})
    if (idx + 1) % 400 == 0:
        print("  已处理", idx + 1)

json.dump(rows, open(OUT, "w"))

N = len(rows)
faces_present = [r for r in rows if r["n"] > 0]
print("\n=== 景别分布 ===")
print("总帧 %d  有人脸帧 %d (%.1f%%)  无人脸帧 %d (%.1f%%)" % (
    N, len(faces_present), len(faces_present) / N * 100,
    N - len(faces_present), (N - len(faces_present)) / N * 100))

from collections import Counter
cc = Counter(r["cls"] for r in rows)
order = ["大特写/特写", "特写", "近景", "中景", "中全景", "全景/远景", "无脸"]
for k in order:
    v = cc.get(k, 0)
    print("  %-12s : %5d  %5.1f%%  %s" % (k, v, v / N * 100, "#" * int(v / N * 120)))

nf = Counter(r["n"] for r in rows)
print("\n=== 同框人数分布 ===")
for k in sorted(nf):
    print("  %d 人 : %5d 帧  %5.1f%%" % (k, nf[k], nf[k] / N * 100))

fh = np.array([r["max_face_h"] for r in faces_present])
print("\n=== 人脸高度占画面比 ===")
print("  均值 %.3f  中位 %.3f  P10 %.3f  P25 %.3f  P75 %.3f  P90 %.3f" % (
    fh.mean(), np.median(fh), *np.percentile(fh, [10, 25, 75, 90])))
cx = np.array([r["cx"] for r in faces_present])
cy = np.array([r["cy"] for r in faces_present])
print("  人物水平中心分布: 左1/3 %.1f%%  中1/3 %.1f%%  右1/3 %.1f%%" % (
    (cx < 1 / 3).mean() * 100, ((cx >= 1 / 3) & (cx < 2 / 3)).mean() * 100, (cx >= 2 / 3).mean() * 100))
print("  人物垂直中心分布: 上1/3 %.1f%%  中1/3 %.1f%%  下1/3 %.1f%%" % (
    (cy < 1 / 3).mean() * 100, ((cy >= 1 / 3) & (cy < 2 / 3)).mean() * 100, (cy >= 2 / 3).mean() * 100))
print("\nwrote", OUT)
