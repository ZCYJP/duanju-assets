"""
底图景别精修：SCRFD 实测人脸位置 → 按 face_ratio 目标做等比裁切/缩放

理由：Qwen-Image-Edit 对 "extreme close-up" 的响应不稳定（实测目标 0.32 实得 0.148），
      但人物位置与像素是确定的 → 用程序化裁切把景别做到目标值，比改 prompt 重试快且准。

裁切原则：
  - 以主脸 bbox 为锚点，按目标比例算人脸在画面中应占的高度
  - 保持人脸水平位置不变（过肩镜的左右构图不能被破坏）
  - 裁切范围不超出原图边界；不够则先放大再裁
"""
import os
import sys
import json
import numpy as np
import cv2

sys.path.insert(0, "/root/.insightface")
sys.path.insert(0, "/ssd2/duanju/drama_1min/sample_15s")
from insightface.app import FaceAnalysis
from shots_15s import SHOTS

BASE = "/ssd2/duanju/drama_1min/sample_15s"
SRC = f"{BASE}/frames_src"
DST = f"{BASE}/frames"
W, H = 480, 832

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
det = app.det_model


def detect_main(img):
    bbs, _ = det.detect(img, max_num=20, metric="default")
    if bbs is None or len(bbs) == 0:
        return None
    areas = (bbs[:, 2] - bbs[:, 0]) * (bbs[:, 3] - bbs[:, 1])
    k = int(np.argmax(areas))
    return bbs[k]


def fit_one(shot):
    name = shot["name"]
    src = f"{SRC}/{name}.png"
    dst = f"{DST}/{name}.png"
    if not os.path.exists(src):
        return {"name": name, "ok": False, "err": "missing src"}
    img = cv2.imread(src)
    if img is None:
        return {"name": name, "ok": False, "err": "decode fail"}
    H0, W0 = img.shape[:2]

    box = detect_main(img)
    if box is None:
        return {"name": name, "ok": False, "err": "no face"}

    fx1, fy1, fx2, fy2 = box[:4]
    fh_px = fy2 - fy1
    cur_ratio = fh_px / H0
    target = shot["face_ratio"]

    # 人脸中心（归一化）
    cx = (fx1 + fx2) / 2.0
    cy = (fy1 + fy2) / 2.0

    # ── 精确几何 ──────────────────────────────────────────────
    # 裁切窗口高 h_crop，resize 到 H 后：脸高_out = fh_px * H / h_crop
    # 要脸高占比 = target → fh_px * H / h_crop = target * H → h_crop = fh_px / target
    # 宽度按输出宽高比取，避免非等比拉伸：new_w = h_crop * (W / H)
    # （原图 832x1248 与输出 480x832 宽高比不同，必须按输出比裁切）
    rel_cx = cx / W0
    rel_cy = cy / H0

    def window(h_crop):
        """给定裁切窗口高，算出窗口矩形（保持脸在原构图相对位置附近）"""
        nw = h_crop * (W / H)
        nh = h_crop
        if nw > W0:            # 宽度受限
            nw = W0
            nh = W0 / (W / H)
        if nh > H0:
            nh = H0
            nw = nh * (W / H)
        # 单人特写：脸置于画面中上部（0.42）；双人镜：保留原相对位置（过肩靠边不能丢）
        want_rel_x = rel_cx
        want_rel_y = 0.42 if shot["framing"] == "single" else rel_cy
        x1 = cx - want_rel_x * nw
        y1 = cy - want_rel_y * nh
        x1 = min(max(x1, 0.0), W0 - nw)
        y1 = min(max(y1, 0.0), H0 - nh)
        return int(round(x1)), int(round(y1)), int(round(nw)), int(round(nh))

    h_crop = fh_px / max(target, 1e-3)
    out = None
    box2 = None
    n2 = 0
    r2 = 0.0
    for it in range(3):
        x1, y1, nw, nh = window(h_crop)
        crop = img[y1:y1 + nh, x1:x1 + nw]
        if crop.size == 0:
            break
        out = cv2.resize(crop, (W, H), interpolation=cv2.INTER_AREA if nw >= W else cv2.INTER_CUBIC)
        box2 = detect_main(out)
        if box2 is None:
            break
        _bb = det.detect(out, max_num=20, metric="default")[0]
        n2 = 0 if _bb is None else len(_bb)
        r2 = (box2[3] - box2[1]) / H
        if abs(r2 - target) <= 0.012:
            break
        # 用实测反推修正窗口高（插值/检测误差补偿）
        h_crop = h_crop * (r2 / max(target, 1e-3))
        h_crop = min(max(h_crop, 40.0), H0)

    if out is None or box2 is None:
        return {"name": name, "ok": False, "err": "crop/face fail"}

    # 双人镜保护：裁切后掉人 → 逐级放大窗口重试，仍不行则回退整图
    want_n = 2 if shot["framing"] == "two_shot" else 1
    if n2 < want_n:
        h2 = h_crop
        for factor in (1.25, 1.55, 2.0):
            h_try = min(h_crop * factor, H0)
            x1, y1, nw, nh = window(h_try)
            c2 = img[y1:y1 + nh, x1:x1 + nw]
            if c2.size == 0:
                continue
            o2 = cv2.resize(c2, (W, H), interpolation=cv2.INTER_AREA if nw >= W else cv2.INTER_CUBIC)
            b2 = detect_main(o2)
            if b2 is None:
                continue
            _bb = det.detect(o2, max_num=20, metric="default")[0]
            nn2 = 0 if _bb is None else len(_bb)
            if nn2 >= want_n:
                out, box2, n2, h2 = o2, b2, nn2, h_try
                r2 = (box2[3] - box2[1]) / H
                break
        else:
            out = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            box2 = detect_main(out)
            if box2 is None:
                return {"name": name, "ok": False, "err": "no face fallback"}
            r2 = (box2[3] - box2[1]) / H
            _bd = det.detect(out, max_num=20, metric="default")[0]
            n2 = 0 if _bd is None else len(_bd)
            h2 = H0

    cv2.imwrite(dst, out)
    return {"name": name, "ok": True, "h_crop": int(h_crop),
            "before": round(cur_ratio, 3), "after": round(r2, 3),
            "target": target, "n": int(n2),
            "err": None if abs(r2 - target) <= 0.055 else "ratio off"}


def main():
    os.makedirs(SRC, exist_ok=True)
    os.makedirs(DST, exist_ok=True)
    res = []
    for shot in SHOTS:
        r = fit_one(shot)
        res.append(r)
        if r["ok"]:
            flag = "OK" if r.get("err") is None else "WARN"
            print("%-24s %-10s 目标%.2f  %.3f→%.3f  h=%d  人=%d  %s" % (
                r["name"], shot["framing"], r["target"], r["before"], r["after"],
                r.get("h_crop", 0), r["n"], flag), flush=True)
        else:
            print("%-24s %-10s FAIL %s" % (r["name"], shot["framing"], r["err"]), flush=True)
    ok = sum(1 for r in res if r["ok"])
    print(f"\n完成 {ok}/{len(res)}")
    json.dump(res, open(f"{BASE}/logs/fit_report.json", "w"), ensure_ascii=False, indent=2, default=float)


if __name__ == "__main__":
    main()
