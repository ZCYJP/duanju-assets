# -*- coding: utf-8 -*-
"""
《深渊》3分钟短剧 — 合成
1. seg 段按镜拼接 (H3 原生语音+口型, 保留原声)
2. 情绪 BGM 合成混入 (低于人声)
3. ASS 字幕烧录 (角色名 + 对白)
"""
import math, os, re, subprocess, sys
import numpy as np
import soundfile as sf

sys.path.insert(0, "/ssd2/short_drama")
from storyboard_char16 import SCENES
from render_char16 import split_line, SEG_DUR

BASE = "/ssd2/short_drama"
SEG = f"{BASE}/comfy_output/char16_drama/seg"
OUT = f"{BASE}/output"
os.makedirs(OUT, exist_ok=True)
FONT_DIR = f"{BASE}/fonts"
SR = 48000

TITLES = ["第一幕 · 决裂", "来者不善", "深渊边缘",
          "第二幕 · 雨夜身影", "亮底牌", "正面交锋",
          "第三幕 · 反转", "收束"]

MOODS = ["tense", "dark", "sad", "rain", "tense", "dark", "triumph", "warm"]

CHORDS = {
    "sad":     [(0, "m"), (8, "M"), (3, "M"), (10, "M")],
    "dark":    [(0, "m"), (10, "M"), (8, "M"), (7, "M")],
    "tense":   [(8, "M"), (3, "M"), (5, "m"), (10, "M")],
    "rain":    [(0, "m"), (5, "m"), (10, "M"), (8, "M")],
    "triumph": [(3, "M"), (10, "M"), (8, "M"), (0, "M")],
    "warm":    [(8, "M"), (3, "M"), (0, "m"), (5, "m")],
}


def build_schedule():
    """与渲染脚本逐段一致的台词调度"""
    sched = []
    for sc in SCENES:
        n = sc["nsegs"]
        pairs = [d.split("：", 1) for d in sc["dialogue"].split("|")]
        chunks = []
        for sp, tx in pairs:
            for ck in split_line(tx):
                chunks.append((sp, ck))
        idx = 0
        for si in range(n):
            rem = n - si
            rc = len(chunks) - idx
            take = min(rc, max(1, math.ceil(rc / rem))) if rc > 0 else 0
            sched.append({"sid": sc["scene_id"], "seg": si, "chunks": chunks[idx:idx + take]})
            idx += take
    return sched


def probe_dur(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except Exception:
        return SEG_DUR


def _write_all_list(paths, lst):
    with open(lst, "w") as f:
        for p in paths:
            f.write(f"file '{p}'\n")
    return lst


# ---------- Step 1: 拼段成镜 ----------
def concat_all():
    segs_per_scene = {}
    for j in build_schedule():
        segs_per_scene.setdefault(j["sid"], []).append(j["seg"])
    scenes = []
    for sid, segs in sorted(segs_per_scene.items()):
        lst = f"{OUT}/char16_list_s{sid}.txt"
        with open(lst, "w") as f:
            for si in sorted(segs):
                f.write(f"file '{SEG}/seg_s{sid}_{si}.mp4'\n")
        out = f"{OUT}/char16_scene_s{sid}.mp4"
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                        "-c", "copy", out], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  镜 {sid}: {len(segs)} 段 -> {probe_dur(out):.1f}s")
        scenes.append(out)
    full = f"{OUT}/char16_full.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i",
                    _write_all_list(scenes, f"{OUT}/char16_list_all.txt"),
                    "-c:v", "libx264", "-crf", "21", "-preset", "fast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", full], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return full


# ---------- Step 2: BGM ----------
def synth_pad(root, ctype, dur, vol=0.09):
    n = int(dur * SR)
    t = np.linspace(0, dur, n, endpoint=False)
    f0 = 220.0 * (2 ** (root / 12.0))
    intervals = [0, 3, 7, 12] if ctype == "m" else [0, 4, 7, 12]
    sig = np.zeros(n)
    for iv in intervals:
        f = f0 * (2 ** (iv / 12))
        for det in (-0.15, 0.15):
            sig += np.sin(2 * np.pi * (f + det) * t) / 4
            sig += 0.3 * np.sin(2 * np.pi * 2 * (f + det) * t) / 4
    fade = min(int(1.2 * SR), n // 3)
    env = np.ones(n)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    env *= (0.85 + 0.15 * np.sin(2 * np.pi * 0.15 * t))
    return (sig * env * vol).astype(np.float32)


def build_bgm(durations):
    pieces = []
    for sid, dur in enumerate(durations, start=1):
        mood = MOODS[sid - 1]
        chords = CHORDS[mood]
        seg = np.zeros(int(dur * SR))
        pos, k, cd = 0.0, 0, 4.0
        while pos < dur:
            seg_d = min(cd, dur - pos)
            r, ct = chords[k % len(chords)]
            pad = synth_pad(r, ct, seg_d)
            p0 = int(pos * SR)
            seg[p0:p0 + len(pad)] += pad[:len(seg) - p0]
            pos += seg_d
            k += 1
        if mood == "dark":       # 低沉鼓点
            n = int(dur * SR)
            beat = 60.0 / 96
            for b in range(int(dur / beat)):
                t0 = int(b * beat * SR)
                tl = int(0.4 * SR)
                if t0 + tl >= n:
                    break
                kt = np.linspace(0, 0.4, tl, endpoint=False)
                s = 1.0 if b % 4 == 0 else (0.6 if b % 2 == 0 else 0.35)
                seg[t0:t0 + tl] += np.sin(2 * np.pi * (110 * np.exp(-kt * 14) + 42) * kt) * np.exp(-kt * 9) * s
        if mood == "rain":       # 雨声底噪
            n = int(dur * SR)
            rng = np.random.default_rng(7)
            rain = rng.normal(0, 0.35, n)
            kern = np.ones(48) / 48
            rain = np.convolve(rain, kern, mode="same")
            seg += rain * 0.05
        pieces.append(seg * 0.5)
        print(f"  BGM 镜{sid}: {dur:.1f}s [{mood}]")
    return np.concatenate(pieces)


# ---------- Step 3: 字幕 ----------
def fmt(t):
    t = max(t, 0)
    return f"{int(t//3600)}:{int((t%3600)//60):02d}:{t%60:05.2f}"


def write_ass(path):
    lines = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1344", "PlayResY: 768",
             "WrapStyle: 0", "ScaledBorderAndShadow: yes", ""]
    lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding")
    lines.append("Style: Title,Noto Sans SC,34,&H0028C8FF,&H000000FF,&H00000000,&H64000000,1,0,0,0,100,100,1,0,1,2.5,1,8,30,30,14,1")
    lines.append("Style: Char,Noto Sans SC,28,&H0028C8FF,&H000000FF,&H00000000,&H64000000,1,0,0,0,100,100,1,0,1,2.2,1,8,30,30,12,1")
    lines.append("Style: Dia,Noto Sans SC,36,&H00FFFFFF,&H000000FF,&H00101010,&H8C000000,1,0,0,0,100,100,0.5,0,1,2.8,1.2,2,40,40,30,1")
    lines += ["", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]

    t = 0.0
    first = True
    for j in build_schedule():
        if j["seg"] == 0 and first:
            lines.append(f"Dialogue: 0,{fmt(t)},{fmt(t+2.4)},Title,,0,0,0,,{TITLES[j['sid']-1]}")
            first = False
        if j["chunks"]:
            spk = j["chunks"][0][0]
            text = "".join(x for _, x in j["chunks"])
            lines.append(f"Dialogue: 0,{fmt(t+0.25)},{fmt(t+SEG_DUR-0.35)},Char,,0,0,0,,{spk}")
            lines.append(f"Dialogue: 0,{fmt(t+0.25)},{fmt(t+SEG_DUR-0.35)},Dia,,0,0,0,,{text}")
        t += SEG_DUR
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ASS written: {path}")
    return path


if __name__ == "__main__":
    print("=" * 60)
    print("  Step 1: 拼段成镜")
    print("=" * 60)
    full = concat_all()
    total = probe_dur(full)
    print(f"  全片时长: {total:.1f}s ({total/60:.2f}min)")

    print("=" * 60)
    print("  Step 2: BGM 混音")
    print("=" * 60)
    durations = [probe_dur(f"{OUT}/char16_scene_s{sid}.mp4")
                 for sid in sorted(set(j["sid"] for j in build_schedule()))]
    bgm = build_bgm(durations)
    need = int(total * SR)
    bgm = np.pad(bgm, (0, need - len(bgm))) if len(bgm) < need else bgm[:need]
    bgm_path = f"{OUT}/char16_bgm.wav"
    sf.write(bgm_path, bgm, SR)
    mixed = f"{OUT}/char16_mixed.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", full, "-i", bgm_path,
                    "-filter_complex",
                    "[0:a]volume=1.0[va];[1:a]volume=0.42,bass=g=2[mu];"
                    "[va][mu]amix=inputs=2:duration=first:dropout_transition=0,alimiter=limit=0.95[a]",
                    "-map", "0:v:0", "-map", "[a]",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", mixed], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("=" * 60)
    print("  Step 3: 字幕烧录")
    print("=" * 60)
    ass_path = write_ass(f"{OUT}/char16_subs.ass")
    final = f"{OUT}/深渊_3分钟成片.mp4"
    subprocess.run(["ffmpeg", "-y", "-i", mixed,
                    "-vf", f"subtitles=filename={ass_path}:fontsdir={FONT_DIR}",
                    "-c:v", "libx264", "-crf", "20", "-preset", "medium",
                    "-pix_fmt", "yuv420p", "-c:a", "copy", final], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  ✅ 完成: {final}  ({os.path.getsize(final)/1024/1024:.1f}MB, {probe_dur(final):.1f}s)")
