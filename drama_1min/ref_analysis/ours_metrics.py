"""我们成片的同口径指标测量（与参考片完全一致的算法）"""
import os
import glob
import json
import subprocess
import numpy as np
import cv2

WORK = "/ssd2/duanju/drama_1min"
VID = os.path.join(WORK, "final/drama_1min_h3.mp4")
FR = "/tmp/refan/ours_metrics"

if not os.path.isdir(FR):
    os.makedirs(FR, exist_ok=True)
    subprocess.run(["/usr/local/bin/ffmpeg", "-v", "error", "-y", "-i", VID,
                    "-vf", "fps=2", "-q:v", "3", os.path.join(FR, "f_%05d.jpg")], check=True)

files = sorted(glob.glob(os.path.join(FR, "*.jpg")))
print("抽帧", len(files))

# ---- 1. 场景切点（与参考片同参数）----
out = os.path.join(FR, "cuts.txt")
subprocess.run(["/usr/local/bin/ffmpeg", "-v", "error", "-i", VID,
                "-vf", "select='gt(scene,0.2)',metadata=print:file=%s" % out,
                "-an", "-f", "null", "-"], check=True)
import re
lines = open(out, encoding="utf-8", errors="ignore").read().splitlines()
cuts = []
for i, l in enumerate(lines):
    m = re.search(r"pts_time:([0-9.]+)", l)
    if m:
        sc = 0.0
        if i + 1 < len(lines) and "scene_score" in lines[i + 1]:
            mm = re.search(r"scene_score=([0-9.]+)", lines[i + 1])
            if mm:
                sc = float(mm.group(1))
        cuts.append((float(m.group(1)), sc))
DUR = 60.0
bounds = [0.0] + [t for t, _ in cuts] + [DUR]
lens = [bounds[i+1] - bounds[i] for i in range(len(bounds)-1)]
lens = [x for x in lens if x > 0.05]
print("\n=== 我们成片 · 镜头结构 ===")
print("镜头数 %d  平均 %.3fs  中位 %.3fs  最长 %.3fs" % (
    len(lens), np.mean(lens), np.median(lens), max(lens)))
print("<=3s 占比 %.1f%%   <=2s %.1f%%   >=4s %.1f%%" % (
    sum(1 for x in lens if x <= 3)/len(lens)*100,
    sum(1 for x in lens if x <= 2)/len(lens)*100,
    sum(1 for x in lens if x >= 4)/len(lens)*100))
print("切换密度 %.3f cuts/s" % (len(lens)/DUR))

# ---- 2. 景别（SCRFD，与参考片同模型同阈值）----
sys_path = None
import sys
sys.path.insert(0, "/root/.insightface")
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
det = app.det_model

rows = []
for i, fp in enumerate(files):
    img = cv2.imread(fp)
    if img is None:
        continue
    H, W = img.shape[:2]
    try:
        bbs, _ = det.detect(img, max_num=20, metric="default")
    except Exception:
        bbs = None
    if bbs is None or len(bbs) == 0:
        rows.append({"f": os.path.basename(fp), "n": 0, "fh": 0.0, "cls": "无脸"})
        continue
    areas = (bbs[:, 2]-bbs[:, 0])*(bbs[:, 3]-bbs[:, 1])
    k = int(np.argmax(areas))
    x1, y1, x2, y2 = bbs[k][:4]
    fh = (y2-y1)/H
    if fh >= 0.28: cls = "大特写/特写"
    elif fh >= 0.18: cls = "特写"
    elif fh >= 0.11: cls = "近景"
    elif fh >= 0.06: cls = "中景"
    elif fh >= 0.02: cls = "中全景"
    else: cls = "全景/远景"
    rows.append({"f": os.path.basename(fp), "n": len(bbs), "fh": float(fh), "cls": cls,
                 "cx": float(((x1+x2)/2)/W), "cy": float(((y1+y2)/2)/H)})

from collections import Counter
N = len(rows)
cc = Counter(r["cls"] for r in rows)
print("\n=== 我们成片 · 景别分布 ===")
for k2 in ["大特写/特写", "特写", "近景", "中景", "中全景", "全景/远景", "无脸"]:
    v = cc.get(k2, 0)
    print("  %-12s : %4d  %5.1f%%  %s" % (k2, v, v/N*100, "#"*int(v/N*140)))
fpr = [r for r in rows if r["n"] > 0]
if fpr:
    fh = np.array([r["fh"] for r in fpr])
    print("人脸高占比 均值 %.3f 中位 %.3f   >0.11(近景+) %.1f%%  >0.06(中景+) %.1f%%" % (
        fh.mean(), np.median(fh), (fh > 0.11).mean()*100, (fh > 0.06).mean()*100))
    nf = Counter(r["n"] for r in rows)
    print("同框人数:", {k2: "%d帧(%.0f%%)" % (v, v/N*100) for k2, v in sorted(nf.items())})

json.dump({"lens": lens, "cuts": len(lens), "scale": dict(cc)},
          open("/tmp/refan/ours_metrics.json", "w"), ensure_ascii=False, indent=2)
print("\nOK")
