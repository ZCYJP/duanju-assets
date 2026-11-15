"""景别分析：insightface SCRFD（det_10g）+ 多进程分片。用 insightface 内部 RetinaFace 封装"""
import os
import sys
import glob
import json
import numpy as np
import cv2

sys.path.insert(0, "/root/.insightface")
from insightface.app import FaceAnalysis

FRAMES = "/tmp/refan/frames"

shard = int(sys.argv[1])
nshard = int(sys.argv[2])

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
det_model = app.det_model


def main():
    files = sorted(glob.glob(os.path.join(FRAMES, "*.jpg")))
    mine = files[shard::nshard]
    rows = []
    for i, fp in enumerate(mine):
        img = cv2.imread(fp)
        if img is None:
            continue
        H, W = img.shape[:2]
        try:
            bboxes, kpss = det_model.detect(img, max_num=20, metric="default")
        except Exception:
            bboxes, kpss = np.zeros((0, 5)), None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if bboxes is None or len(bboxes) == 0:
            rows.append({"f": os.path.basename(fp), "n": 0, "fh": 0.0, "cls": "无脸",
                         "bright": float(gray.mean())})
        else:
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            k = int(np.argmax(areas))
            x1, y1, x2, y2 = bboxes[k][:4]
            fh = (y2 - y1) / H
            fw = (x2 - x1) / W
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
            rows.append({"f": os.path.basename(fp), "n": len(bboxes), "fh": float(fh),
                         "fw": float(fw), "cls": cls,
                         "cx": float(((x1 + x2) / 2) / W), "cy": float(((y1 + y2) / 2) / H),
                         "bright": float(gray.mean())})
        if (i + 1) % 100 == 0:
            print("  shard%d %d/%d" % (shard, i + 1, len(mine)), flush=True)
    json.dump(rows, open("/tmp/refan/shards/scale_%d.json" % shard, "w"))
    print("shard%d done %d" % (shard, len(rows)))


main()
