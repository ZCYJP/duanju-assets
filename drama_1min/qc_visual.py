#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成片视觉体检（程序化，替代人工看图）：
  1. 每帧基本信息：亮度、对比度、饱和度、是否全黑/过曝/冻结
  2. 相邻帧差异（检测画面是否卡死/静止）
  3. 跨镜人物服装一致性：把每镜人物区域（上 2/3）做主色提取，
     同角色不同镜的主色距离过大则告警
  4. 帧冻结检测：连续帧像素差 < 阈值
"""
import os, sys, json, subprocess, tempfile
import numpy as np
from PIL import Image

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "/ssd2/duanju/drama_1min/final/drama_1min_final.mp4"
TMP = tempfile.mkdtemp(prefix="visqc_")
FPS = 16

# 12 镜在成片中的起始秒（片头 1.0s + 每镜 5.0s，末镜 4.0s）
SHOT_START = [1.0 + i * 5.0 for i in range(12)]
SHOT_CHAR = ["F01", "F03", "M03", "F08", "F01", "M01",
             "M01", "M03", "F03", "F08", "F01", "M01"]


def extract_frames():
    for f in os.listdir(TMP):
        os.remove(os.path.join(TMP, f))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", VIDEO,
                    "-vf", f"fps={FPS}", os.path.join(TMP, "f_%04d.png")], check=True)
    return sorted(os.path.join(TMP, f) for f in os.listdir(TMP) if f.endswith(".png"))


def analyze(path):
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(np.float32) / 255.0
    gray = a.mean(axis=2)
    hsv_max = a.max(axis=2)
    hsv_min = a.min(axis=2)
    sat = np.where(hsv_max > 0, (hsv_max - hsv_min) / (hsv_max + 1e-6), 0)
    return a, gray, float(gray.mean()), float(gray.std()), float(sat.mean())


def main():
    frames = extract_frames()
    print(f"总帧数: {len(frames)} @ {FPS}fps = {len(frames)/FPS:.2f}s")

    # ---- 1. 全局统计 ----
    rows = []
    prev = None
    freezes = []
    for i, p in enumerate(frames):
        a, gray, mu, sd, sat = analyze(p)
        diff = None
        if prev is not None:
            diff = float(np.mean(np.abs(a - prev)))
        prev = a
        if diff is not None and diff < 0.002:
            freezes.append((i, diff))
        rows.append({"i": i, "t": round(i / FPS, 3), "mu": round(mu, 4),
                     "sd": round(sd, 4), "sat": round(sat, 4), "diff": diff})

    mus = np.array([r["mu"] for r in rows])
    sds = np.array([r["sd"] for r in rows])
    sats = np.array([r["sat"] for r in rows])
    print(f"亮度 mean={mus.mean():.3f} min={mus.min():.3f} max={mus.max():.3f}")
    print(f"对比度 mean={sds.mean():.3f} min={sds.min():.3f}")
    print(f"饱和度 mean={sats.mean():.3f}")
    black = [r["i"] for r in rows if r["mu"] < 0.02]
    over = [r["i"] for r in rows if r["mu"] > 0.97]
    flat = [r["i"] for r in rows if r["sd"] < 0.03]
    print(f"全黑帧(<0.02): {black if black else '无'}")
    print(f"过曝帧(>0.97): {over if over else '无'}")
    print(f"低对比帧(<0.03): {flat if flat else '无'}")
    print(f"冻结帧(相邻差<0.002): {len(freezes)} 帧" +
          (f" -> {[i for i,_ in freezes[:10]]}" if freezes else ""))

    # ---- 2. 跨镜人物区域主色一致性 ----
    print("\n=== 跨镜人物服装主色一致性 ===")
    # 每个角色取各镜中间偏上区域（人物躯干区），避开字幕（下部）
    char_frames = {}
    for idx, start in enumerate(SHOT_START):
        # 取镜内 2.0s 处（避开转场与字幕起始），人物躯干大约在画面 30%-60% 高度
        fi = min(int((start + 2.0) * FPS), len(frames) - 1)
        a, gray, mu, sd, sat = analyze(frames[fi])
        h = a.shape[0]
        torso = a[int(h * 0.30):int(h * 0.62), int(a.shape[1] * 0.20):int(a.shape[1] * 0.80)]
        # 量化主色（4bit/channel = 4096 bins），取前 5 主色
        q = (torso * 15).astype(np.int32)
        qflat = q[:, :, 0] * 256 + q[:, :, 1] * 16 + q[:, :, 2]
        bins, counts = np.unique(qflat, return_counts=True)
        order = np.argsort(-counts)[:5]
        tops = []
        for o in order:
            b = int(bins[o])
            tops.append(((b // 256) / 15.0, ((b // 16) % 16) / 15.0, (b % 16) / 15.0,
                         float(counts[o] / qflat.size)))
        char_frames.setdefault(SHOT_CHAR[idx], []).append((idx, tops))

    issues = []
    for cid, lst in sorted(char_frames.items()):
        if len(lst) < 2:
            continue
        print(f"\n  角色 {cid} ({len(lst)} 镜):")
        for idx, tops in lst:
            desc = " ".join(f"rgb({r:.2f},{g:.2f},{b:.2f}){w*100:.0f}%" for r, g, b, w in tops[:3])
            print(f"    镜{idx+1:02d}: {desc}")
        # 两两比较主色集合的最小距离
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                d = min(np.linalg.norm(np.array(t[:3]) - np.array(u[:3]))
                        for t in lst[i][1] for u in lst[j][1])
                if d > 0.35:
                    issues.append((cid, lst[i][0] + 1, lst[j][0] + 1, round(float(d), 3)))
                    print(f"    ⚠️ 镜{lst[i][0]+1:02d} vs 镜{lst[j][0]+1:02d} 主色距离 {d:.3f} (阈值0.35)")

    print(f"\n服装一致性告警: {len(issues)} 处")
    for cid, a, b, d in issues:
        print(f"  {cid}: 镜{a:02d} vs 镜{b:02d} Δ={d}")

    report = {"frames": len(rows), "fps": FPS, "duration": round(len(frames) / FPS, 3),
              "black": [int(x) for x in black], "over": [int(x) for x in over],
              "flat": [int(x) for x in flat],
              "freeze_count": len(freezes),
              "brightness_mean": round(float(mus.mean()), 4),
              "contrast_mean": round(float(sds.mean()), 4),
              "saturation_mean": round(float(sats.mean()), 4),
              "costume_issues": [[str(c), int(a), int(b), float(d)] for c, a, b, d in issues]}
    with open("/ssd2/duanju/drama_1min/final/visual_qc.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n报告: /ssd2/duanju/drama_1min/final/visual_qc.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
