"""字幕样式分析：easyocr 检测字幕行位置/字号/颜色 + 判断有无大字标题层"""
import os
import glob
import json
import numpy as np
import cv2
import easyocr

FRAMES = "/tmp/refan/frames_hi"
OUT = "/tmp/refan/subtitle_style.json"

reader = easyocr.Reader(["ch_sim", "en"], gpu=False, verbose=False)
files = sorted(glob.glob(os.path.join(FRAMES, "*.jpg")))
print("帧数", len(files))

rows = []
for i, fp in enumerate(files):
    img = cv2.imread(fp)
    if img is None:
        continue
    H, W = img.shape[:2]
    res = reader.readtext(img, detail=1, paragraph=False)
    if not res:
        rows.append({"f": os.path.basename(fp), "n": 0})
    else:
        items = []
        for box, txt, conf in res:
            if conf < 0.3:
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            # 取文本主体颜色（中心区域中位色）
            cx0, cx1 = int(max(0, x1)), int(min(W, x2))
            cy0, cy1 = int(max(0, y1)), int(min(H, y2))
            crop = img[cy0:cy1, cx0:cx1]
            if crop.size == 0:
                bgr = (255, 255, 255)
            else:
                bgr = tuple(int(v) for v in np.median(crop.reshape(-1, 3), axis=0))
            items.append({"t": txt, "conf": round(float(conf), 2),
                          "y1": float(y1 / H), "y2": float(y2 / H),
                          "x1": float(x1 / W), "x2": float(x2 / W),
                          "h_px": float(y2 - y1), "h_rel": float((y2 - y1) / H),
                          "bgr": bgr})
        rows.append({"f": os.path.basename(fp), "n": len(items), "items": items})
    if (i + 1) % 50 == 0:
        print("  %d/%d" % (i + 1, len(files)), flush=True)

json.dump(rows, open(OUT, "w"), ensure_ascii=False)

# ---- 统计 ----
allitems = [it for r in rows for it in r.get("items", [])]
print("\n=== OCR 统计 ===")
print("总帧 %d  有文字帧 %d (%.1f%%)  总文本块 %d" % (
    len(rows), sum(1 for r in rows if r["n"] > 0),
    sum(1 for r in rows if r["n"] > 0) / len(rows) * 100, len(allitems)))

# 字幕带：y 中心在 0.70~0.98
sub = [it for it in allitems if it["y1"] > 0.60]
print("\n=== 字幕带（y1>0.60）===")
print("文本块 %d，占全部 %.1f%%" % (len(sub), len(sub) / max(len(allitems), 1) * 100))
if sub:
    ys = np.array([it["y1"] for it in sub])
    ye = np.array([it["y2"] for it in sub])
    hh = np.array([it["h_px"] for it in sub])
    print("  y1: 中位 %.3f  P10 %.3f  P90 %.3f" % (np.median(ys), np.percentile(ys, 10), np.percentile(ys, 90)))
    print("  y2: 中位 %.3f" % np.median(ye))
    print("  字号(像素高, 640高画面): 中位 %.1f  P25 %.1f  P75 %.1f  → 占画面高 %.1f%%" % (
        np.median(hh), np.percentile(hh, 25), np.percentile(hh, 75), np.median(hh) / 640 * 100))
    print("  水平跨度 x1 中位 %.3f  x2 中位 %.3f" % (
        np.median([it["x1"] for it in sub]), np.median([it["x2"] for it in sub])))

print("\n=== 非字幕带文字（y1<=0.60）—— 大字标题/贴纸/水印 ===")
other = [it for it in allitems if it["y1"] <= 0.60]
print("文本块 %d" % len(other))
if other:
    oy = np.array([it["y1"] for it in other])
    oh = np.array([it["h_px"] for it in other])
    print("  y1 分布: 上1/3 %d, 中1/3 %d, 下1/3 %d" % (
        (oy < 0.33).sum(), ((oy >= 0.33) & (oy < 0.66)).sum(), (oy >= 0.66).sum()))
    print("  字号像素高 中位 %.1f，最大 %.1f（画面高640）" % (np.median(oh), oh.max()))
    print("  大字块(>40px高) %d 个" % (oh > 40).sum())
    for it in sorted(other, key=lambda x: -x["h_px"])[:15]:
        print("   y=%.2f h=%.0fpx '%s'" % (it["y1"], it["h_px"], it["t"][:40]))

print("\n=== 字幕颜色（取带内块中心中位色，BGR）===")
if sub:
    b = np.array([it["bgr"] for it in sub])
    print("  中位 BGR %s → RGB #%02X%02X%02X" % (
        tuple(int(v) for v in np.median(b, axis=0)),
        int(np.median(b[:, 2])), int(np.median(b[:, 1])), int(np.median(b[:, 0]))))

print("\n=== 样例（前 25 帧的识别文本）===")
cnt = 0
for r in rows:
    for it in r.get("items", []):
        if cnt < 40:
            print("  %s y=%.2f h=%.0f '%s'" % (r["f"], it["y1"], it["h_px"], it["t"][:44]))
            cnt += 1
print("\nwrote", OUT)
