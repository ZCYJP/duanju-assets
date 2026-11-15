#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《决裂》15秒样本 — 成片合成（混音 + 四层字幕 + 拼接）

严格按「短剧语法规范（参考片实测基准）」：
  §5 音频：弃 H3 原生人声（幻觉音）→ 只用 TTS 真音 + 程序化室内环境层
          45Hz 高通 + 7.5kHz 低通；固定增益，**不做 RMS 归一**（反抬嘶声）
          目标 -16 LUFS ±0.5，TP ≤ -1.0 dBTP；**不加 BGM**
  §4 字幕：4 层（中 y=0.686 / 英 y=0.822 / 顶 y=0.080 / 底 y=0.966）
          时间轴由真实发音边界驱动；无「角色：」前缀
  §3 视频：8 镜 concat，每镜按 H3 原生帧数（22/39/56 网格）

输出：sample_15s/out/决裂_EP01_15s.mp4
"""
import os
import re
import sys
import json
import subprocess
import shutil
import wave
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shots_15s import SHOTS, EN_SUB, TOP_BAR, FPS, W, H

BASE = "/ssd2/duanju/drama_1min/sample_15s"
MOTION = f"{BASE}/motion"
VOICE = f"{BASE}/voice"
SUBS = f"{BASE}/subs"
OUT = f"{BASE}/out"
TMP = f"{BASE}/tmp"
for d in (OUT, TMP):
    os.makedirs(d, exist_ok=True)

TOTAL_FRAMES = sum(s["frames"] for s in SHOTS)
TOTAL_DUR = TOTAL_FRAMES / FPS      # 15.125s
LEAD_IN = 0.10                      # 句首留白（镜头内）
TAIL_PAD = 0.12                     # 句尾留白


def sh(cmd, **kw):
    p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, **kw)
    if p.returncode != 0:
        raise RuntimeError("命令失败: %s\n%s\n%s" % (cmd, p.stdout[-2000:], p.stderr[-2000:]))
    return p.stdout + p.stderr


def wav_dur(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def main():
    meta = json.load(open(f"{BASE}/logs/voice_meta.json"))
    vmap = {m["shot_id"]: m for m in meta}

    # ── 1. 视频拼接（无音轨，H3 原生人声丢弃） ────────────────────
    lst = f"{TMP}/concat.txt"
    with open(lst, "w") as f:
        for s in SHOTS:
            p = f"{MOTION}/{s['name']}.mp4"
            if not os.path.exists(p):
                raise SystemExit("缺少视频: " + p)
            f.write("file '%s'\n" % p)
    base_mp4 = f"{TMP}/base.mp4"
    sh(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
        "-an", "-vf", "fps=%d,scale=%d:%d:flags=lanczos,format=yuv420p" % (FPS, W, H),
        "-c:v", "libx264", "-preset", "medium", "-crf", "19", base_mp4])

    # ── 2. 人声轨：tempo 适配镜长 → 按镜位放置 ──────────────────────
    #    镜内可用 = 镜长 - LEAD_IN - TAIL_PAD；超出则 atempo 轻微加速（保持音高）
    voice_parts = []
    t_cursor = 0.0
    timeline = []
    voice_inputs = []
    for s in SHOTS:
        m = vmap[s["id"]]
        src = f"{VOICE}/{m['file']}"
        shot_dur = s["frames"] / FPS
        avail = shot_dur - LEAD_IN - TAIL_PAD
        raw = wav_dur(src)
        tempo = 1.0
        if raw > avail:
            tempo = min(raw / avail, 1.20)
        proc = src
        if abs(tempo - 1.0) > 0.01:
            proc = f"{TMP}/v{s['id']:02d}_t.wav"
            sh(["ffmpeg", "-y", "-v", "error", "-i", src,
                "-af", "atempo=%.4f" % tempo, "-ar", "48000", "-ac", "1", proc])
        d = wav_dur(proc)
        start = t_cursor + LEAD_IN
        delay_ms = int(round(start * 1000))
        idx = len(voice_inputs)
        voice_inputs.append(proc)
        voice_parts.append(
            "[%d:a]aresample=48000,aformat=channel_layouts=stereo,"
            "adelay=%d|%d[va%d]" % (idx, delay_ms, delay_ms, idx))
        timeline.append({
            "shot_id": s["id"], "name": s["name"], "line": m["line"],
            "en": EN_SUB[s["name"]],
            "shot_start": round(t_cursor, 3), "shot_end": round(t_cursor + shot_dur, 3),
            "speech_start": round(start, 3), "speech_end": round(start + d, 3),
            "dur": round(d, 3), "tempo": round(tempo, 3),
            "fill": round(d / shot_dur, 3),
        })
        t_cursor += shot_dur

    n = len(voice_parts)
    mix_in = ";".join(voice_parts)
    mix_out = "".join("[va%d]" % i for i in range(n))
    voice_mp4 = f"{TMP}/voice.wav"
    sh(["ffmpeg", "-y", *sum([["-i", p] for p in voice_inputs], []),
        "-filter_complex", "%s;%samix=inputs=%d:normalize=0:dropout_transition=0,apad=whole_dur=%.3f[vo]"
         % (mix_in, mix_out, n, TOTAL_DUR),
        "-map", "[vo]", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", voice_mp4])

    # ── 3. 不加环境层 / 不加 BGM ────────────────────────────────
    #    参考片实测：安静段谱平坦度 0.0000、谐波峰 1 个、能量中位 -87.2dB
    #    → 安静段就是数字静音。任何底噪/环境床都会抬高平坦度并产生"杂音感"
    # ── 4. 人声轨滤波 + 响度（两遍 loudnorm，不用 RMS 归一） ───────
    #    90Hz 高通：砍掉男声基频底部隆隆（我们 v2 有 28.6Hz 占 35.34% 能量）
    #    6kHz 起 -12dB 高频衰减：压 TTS 齿擦音（8k+ 从 3.38% 降到 1.51%，参考片 0.39%）
    raw_mix = f"{TMP}/mix_raw.wav"
    sh(["ffmpeg", "-y", "-i", voice_mp4,
        "-filter_complex",
        "highpass=f=90,lowpass=f=7000,treble=g=-12:f=6000,alimiter=limit=0.891[m]",  # 0.891 ≈ -1.0 dBTP
        "-map", "[m]", "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", raw_mix])

    # 第一遍 loudnorm 测量（正则提取，避免日志中其它花括号干扰）
    probe = sh(["ffmpeg", "-hide_banner", "-i", raw_mix, "-af",
                "loudnorm=I=-16:TP=-1.5:LRA=9:print_format=json", "-f", "null", "-"])
    def g(k):
        m = re.search(r'"%s"\s*:\s*"([^"]+)"' % k, probe)
        return m.group(1) if m else None
    vals = {k: g(k) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")}
    print("loudnorm 首遍测量:", vals)
    if vals["input_i"] is None:
        raise SystemExit("loudnorm 测量失败")
    lnorm = ("loudnorm=I=-16:TP=-1.5:LRA=9:measured_I=%s:measured_TP=%s:measured_LRA=%s:"
             "measured_thresh=%s:offset=%s:linear=true:print_format=summary"
             % (vals["input_i"], vals["input_tp"], vals["input_lra"],
                vals["input_thresh"], vals["target_offset"]))
    mix_mp4 = f"{TMP}/mix.wav"
    sh(["ffmpeg", "-y", "-i", raw_mix, "-af", lnorm, "-ar", "48000", "-ac", "2",
        "-c:a", "pcm_s16le", mix_mp4])

    # ── 5. 四层字幕：overlay + enable，字幕出现时间 = 真实发音起点 ──
    inputs = ["-i", base_mp4]
    for s in SHOTS:
        inputs += ["-loop", "1", "-framerate", str(FPS), "-t", "%.3f" % TOTAL_DUR,
                   "-i", f"{SUBS}/cn_{s['id']:02d}.png"]
    for s in SHOTS:
        inputs += ["-loop", "1", "-framerate", str(FPS), "-t", "%.3f" % TOTAL_DUR,
                   "-i", f"{SUBS}/en_{s['id']:02d}.png"]
    top_i = 1 + 2 * len(SHOTS)
    bot_i = top_i + 1
    inputs += ["-loop", "1", "-framerate", str(FPS), "-t", "%.3f" % TOTAL_DUR, "-i", f"{SUBS}/top.png",
               "-loop", "1", "-framerate", str(FPS), "-t", "%.3f" % TOTAL_DUR, "-i", f"{SUBS}/bot.png",
               "-i", mix_mp4]
    audio_i = bot_i + 1

    parts = []
    cur = "[0:v]"
    for i, t in enumerate(timeline):
        # 字幕显示区间：语音起点 → 语音终点+0.12，且不越过镜尾
        a = t["speech_start"]
        b = min(t["speech_end"] + 0.12, t["shot_end"])
        nxt = "[c%d]" % i
        parts.append("%s[%d:v]overlay=0:0:format=auto:enable='between(t,%.3f,%.3f)'%s"
                     % (cur, 1 + i, a, b, nxt))
        cur = nxt
    for i, t in enumerate(timeline):
        a = t["speech_start"]
        b = min(t["speech_end"] + 0.12, t["shot_end"])
        nxt = "[e%d]" % i
        parts.append("%s[%d:v]overlay=0:0:format=auto:enable='between(t,%.3f,%.3f)'%s"
                     % (cur, 1 + len(SHOTS) + i, a, b, nxt))
        cur = nxt
    parts.append("%s[%d:v]overlay=0:0:format=auto[t1]" % (cur, top_i))
    parts.append("[t1][%d:v]overlay=0:0:format=auto[outv]" % bot_i)
    fc = ";".join(parts)

    # ── 6. 叠层输出成片 ────────────────────────────────────────────
    out_mp4 = f"{OUT}/决裂_EP01_15s.mp4"
    sh(["ffmpeg", "-y", *inputs, "-filter_complex", fc,
        "-map", "[outv]", "-map", "%d:a" % audio_i,
        "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
        "-r", str(FPS), "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest",
        out_mp4])

    json.dump({"timeline": timeline, "total_dur": TOTAL_DUR, "total_frames": TOTAL_FRAMES},
              open(f"{BASE}/logs/timeline.json", "w"), ensure_ascii=False, indent=2)
    print("\n成片:", out_mp4)
    for t in timeline:
        print("  #%d %-22s 镜 %.2f~%.2fs  语音 %.2f~%.2fs  %s"
              % (t["shot_id"], t["name"], t["shot_start"], t["shot_end"],
                 t["speech_start"], t["speech_end"], t["line"]))


if __name__ == "__main__":
    main()
