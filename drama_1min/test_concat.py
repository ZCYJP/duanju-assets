#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""视频拼接：concat 滤镜（帧级精确），替代 concat demuxer（会丢帧）"""
import os, sys, subprocess, glob

BASE = "/ssd2/duanju/drama_1min"
WORK = f"{BASE}/compose_h3_work"
FFMPEG = "/ssd2/miniconda3/envs/videogen/bin/ffmpeg"
FFPROBE = "/ssd2/miniconda3/envs/videogen/bin/ffprobe"
FPS = 24

segs = [f"{WORK}/intro.mp4"] + sorted(glob.glob(f"{WORK}/v_*.mp4"))
print(f"输入 {len(segs)} 段")

cmd = [FFMPEG, "-y"]
for p in segs:
    cmd += ["-i", p]
fc = []
for i in range(len(segs)):
    fc.append(f"[{i}:v]fps={FPS},format=yuv420p,setpts=PTS-STARTPTS[v{i}]")
chain = "".join(f"[v{i}]" for i in range(len(segs)))
fc.append(f"{chain}concat=n={len(segs)}:v=1:a=0[vout]")
cmd += ["-filter_complex", ";".join(fc), "-map", "[vout]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(FPS),
        f"{WORK}/video_only2.mp4"]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr[-3000:]); raise SystemExit(1)

n = subprocess.run([FFPROBE, "-v", "error", "-count_frames", "-select_streams", "v:0",
                    "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
                    f"{WORK}/video_only2.mp4"], capture_output=True, text=True).stdout.strip()
d = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                    "-of", "csv=p=0", f"{WORK}/video_only2.mp4"],
                   capture_output=True, text=True).stdout.strip()
print(f"concat 滤镜结果: {n} 帧  {d}s  (期望 1446 帧 / 60.25s)")
