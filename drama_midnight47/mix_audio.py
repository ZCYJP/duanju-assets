#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H3 版音频混音引擎 —— 全 Python/numpy 样本级处理，FFmpeg 只做解码与封装。

为什么不用 ffmpeg 滤镜链（2026-11-15 实测踩坑）：
  1. 同一 filter 输出被复用（sidechaincompress + apad 双接）→ ffmpeg 只喂第一路，配音丢失
  2. adelay 推迟的是「整段 wav」（含 TTS 句首静音/换气），不是「发音起点」
     → 字幕比真实发音早 0.26~0.90s
  3. adelay 后接 apad/atrim 会把时基整体推歪（实测语音从 0.31s 被拉到 0.05s）
     → 全部台词滑到镜尾（s04 出现在 3.54s）

本模块的做法：
  - 解码原生音轨 → numpy float32
  - 逐样本检测配音「真实发声点」（10ms 窗能量 + 连续 3 窗确认）
  - 精确裁剪/补静音，使发声点落在镜内 PAD_HEAD 处（样本级，零误差）
  - 确定性增益（实测 RMS → 目标 RMS），不用 loudnorm（短音频会过度衰减）
  - 基于配音瞬时包络的环境音 duck（纯 numpy，可控）
  - 输出整片 60.25s 立体声 wav + 逐句时间轴
"""
import os
import sys
import json
import wave
import subprocess

import numpy as np

SR = 44100
FFMPEG = "/ssd2/miniconda3/envs/videogen/bin/ffmpeg"

# 混音参数
AMBIENT_TARGET_RMS = -34.0   # 环境音目标 RMS(dBFS)（垫底，比人声低 14dB）
VOICE_TARGET_RMS = -20.0     # 配音目标 RMS(dBFS)
DUCK_AMOUNT = 0.25           # 配音期间环境音压到 25%
DUCK_ATTACK_MS = 15
DUCK_RELEASE_MS = 260
VOICE_ACTIVE_FLOOR = 0.02    # 包络归一化后低于此值视为无声


def decode_audio(path, sr=SR, channels=2):
    """用 ffmpeg 解码任意音频/视频文件的音轨为 numpy float32 (n, ch)"""
    cmd = [FFMPEG, "-v", "error", "-i", path, "-vn",
           "-f", "f32le", "-ac", str(channels), "-ar", str(sr), "-"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"解码失败 {path}: {r.stderr[-500:]!r}")
    d = np.frombuffer(r.stdout, dtype=np.float32)
    return d.reshape(-1, channels)


def read_wav(path):
    """读 wav → (n, ch) float32"""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        n = w.getnframes()
        raw = w.readframes(n)
    d = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        d = d.reshape(-1, ch)
    else:
        d = d.reshape(-1, 1)
    return d, sr


def write_wav(path, data):
    """写 wav（float32 (n, ch) → pcm_s16le）"""
    pcm = np.clip(data, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def rms_db(x):
    if x.size == 0:
        return -120.0
    return 20.0 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-9))


def envelope(mono, sr=SR, win_ms=10):
    """10ms 窗 RMS 包络"""
    win = max(1, int(sr * win_ms / 1000))
    n = len(mono) // win
    if n == 0:
        return np.zeros(0, dtype=np.float32), win
    env = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = mono[i * win:(i + 1) * win]
        env[i] = np.sqrt(np.mean(seg ** 2))
    return env, win


def speech_bounds(mono, sr=SR):
    """
    检测配音的真实发声点 / 收声点（样本索引）。
    阈值 = 峰值 8%（保守，避免切掉辅音起始），且要求连续 3 窗（30ms）确认。
    """
    env, win = envelope(mono, sr)
    if env.size == 0:
        return 0, 0
    if env.size < 3:
        # 音频过短：退化为峰值点
        pk = int(np.argmax(env))
        return pk * win, min((pk + 1) * win, len(mono))
    peak = float(env.max())
    if peak <= 1e-6:
        return 0, 0
    thr = peak * 0.08
    start = None
    for i in range(env.size):
        if i + 2 < env.size and env[i] >= thr and env[i + 1] >= thr and env[i + 2] >= thr:
            start = i
            break
    if start is None:
        start = int(np.argmax(env))
    end = env.size
    for i in range(env.size - 1, -1, -1):
        if env[i] >= thr:
            end = i + 1
            break
    return start * win, end * win


def duck_gain_curve(voice_mono, sr=SR):
    """基于配音包络生成环境音增益曲线（attack 快 / release 慢），样本级"""
    win = max(1, int(sr * 0.01))
    n = len(voice_mono) // win
    if n == 0:
        return np.ones(0, dtype=np.float32)
    env = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = voice_mono[i * win:(i + 1) * win]
        env[i] = np.sqrt(np.mean(seg ** 2))
    peak = float(env.max())
    if peak <= 1e-6:
        return np.ones(len(voice_mono), dtype=np.float32)
    norm = env / peak
    act = np.clip((norm - VOICE_ACTIVE_FLOOR) / 0.25, 0.0, 1.0)
    a = float(np.exp(-10.0 / max(1.0, DUCK_ATTACK_MS)))
    r = float(np.exp(-10.0 / max(1.0, DUCK_RELEASE_MS)))
    g = np.empty(n, dtype=np.float32)
    cur = 1.0
    for i in range(n):
        tgt = 1.0 - DUCK_AMOUNT * float(act[i])
        if tgt < cur:
            cur = a * cur + (1.0 - a) * tgt
        else:
            cur = r * cur + (1.0 - r) * tgt
        g[i] = cur
    idx = np.minimum(np.arange(len(voice_mono)) // win, n - 1)
    return g[idx].astype(np.float32)


def fit_length(data, n_samples):
    """把 (n, ch) 截断或补零到 n_samples"""
    if len(data) >= n_samples:
        return data[:n_samples].copy()
    out = np.zeros((n_samples, data.shape[1]), dtype=np.float32)
    out[:len(data)] = data
    return out


def mix_shot(motion_path, voice_path, seg_dur, pad_head):
    """
    混合单镜：原生环境音 + TTS 配音（duck）。
    返回 (mixed (n,2), meta dict)
      meta.speech_offset   配音在镜内的发声点（秒）
      meta.speech_end      配音收声点在镜内的位置（秒）
    """
    n_seg = int(round(seg_dur * SR))
    amb = fit_length(decode_audio(motion_path), n_seg)

    if voice_path is None or not os.path.exists(voice_path):
        a_rms = rms_db(amb)
        g = 10 ** ((AMBIENT_TARGET_RMS - a_rms) / 20.0)
        mixed = amb * g
        return mixed, {"speech_offset": None, "speech_end": None,
                       "amb_rms": a_rms, "voice_rms": None,
                       "amb_gain_db": 20 * np.log10(max(g, 1e-9)),
                       "voice_gain_db": None}

    # --- 配音：读入 → 定位真实发声点 → 精确放置 ---
    vraw, vsr = read_wav(voice_path)
    vmono = vraw.mean(axis=1) if vraw.shape[1] > 1 else vraw[:, 0]
    if vsr != SR:
        # 线性重采样（本项目 TTS 均为 44100，兜底）
        n_out = int(len(vmono) * SR / vsr)
        vmono = np.interp(np.linspace(0, len(vmono) - 1, n_out),
                          np.arange(len(vmono)), vmono).astype(np.float32)
    i_start, i_end = speech_bounds(vmono)
    # 语音段电平（发声点 → 收声点），用于确定性增益
    speech_seg = vmono[i_start:i_end] if i_end > i_start else vmono
    v_rms = rms_db(speech_seg)

    # 精确放置：目标 = 发声点落在镜内 pad_head 处
    #   cut     = 裁掉的句首静音样本数（可能为 0）
    #   prepend = 补的前置静音样本数（可能为 0）
    target = int(round(pad_head * SR))
    cut = max(0, i_start - target)
    prepend = target - (i_start - cut)          # = target - (i_start - cut)
    voice = vmono[cut:]
    if prepend > 0:
        voice = np.concatenate([np.zeros(prepend, dtype=np.float32), voice])
    n_voice = n_seg
    voice = fit_length(voice.reshape(-1, 1), n_voice)
    voice_mono = voice[:, 0]

    # 确定性增益
    v_gain = 10 ** ((VOICE_TARGET_RMS - v_rms) / 20.0)
    a_rms = rms_db(amb)
    a_gain = 10 ** ((AMBIENT_TARGET_RMS - a_rms) / 20.0)

    voiced = voice_mono * v_gain
    duck = duck_gain_curve(voiced)
    n_duck = min(len(duck), n_seg)
    duck = np.concatenate([duck[:n_duck], np.ones(n_seg - n_duck, dtype=np.float32)])
    mixed = amb * a_gain * duck.reshape(-1, 1) + voiced.reshape(-1, 1)

    speech_offset = pad_head
    speech_end = pad_head + max(0.0, (i_end - i_start) / SR)
    meta = {
        "speech_offset": round(speech_offset, 4),
        "speech_end": round(min(speech_end, seg_dur), 4),
        "amb_rms": round(a_rms, 2),
        "voice_rms": round(v_rms, 2),
        "amb_gain_db": round(20 * np.log10(max(a_gain, 1e-9)), 2),
        "voice_gain_db": round(20 * np.log10(max(v_gain, 1e-9)), 2),
        "voice_file_dur": round(len(vmono) / SR, 3),
        "raw_speech_start": round(i_start / SR, 3),
        "raw_speech_end": round(i_end / SR, 3),
    }
    return mixed, meta


def build_full_mix(shots, motion_dir, voice_dir, intro_dur, out_wav, timeline_path):
    """
    shots: [{id,name,dur,seg_dur,speaker,text,voice_path}]
    输出整片混音 wav + 时间轴 json
    """
    pad_head = float(os.environ.get("PAD_HEAD", "0.40"))
    total = intro_dur + sum(s["seg_dur"] for s in shots)
    n_total = int(round(total * SR))
    full = np.zeros((n_total, 2), dtype=np.float32)

    cursor = intro_dur
    rows = []
    for s in shots:
        mixed, meta = mix_shot(f"{motion_dir}/{s['name']}.mp4",
                               s.get("voice_path"), s["seg_dur"], pad_head)
        n_seg = len(mixed)
        off = int(round(cursor * SR))
        full[off:off + n_seg] = mixed
        shot_start = cursor
        cursor += s["seg_dur"]
        if meta["speech_offset"] is not None:
            sub_start = round(shot_start + meta["speech_offset"], 3)
            sub_end = round(shot_start + meta["speech_end"], 3)
        else:
            sub_start = sub_end = None
        rows.append({
            "shot": s["name"], "speaker": s.get("speaker"), "text": s.get("text"),
            "start": round(shot_start, 3), "end": round(cursor, 3),
            "sub_start": sub_start, "sub_end": sub_end,
            "seg_dur": round(s["seg_dur"], 4), **meta,
        })
        # 无台词镜头（speech_offset=None）也要能打印 —— 纯环境音镜
        _so = "无" if meta["speech_offset"] is None else f"{meta['speech_offset']}"
        _se = "无" if meta["speech_end"] is None else f"{meta['speech_end']}"
        _vr = "无" if meta["voice_rms"] is None else f"{meta['voice_rms']}"
        _vg = "无" if meta["voice_gain_db"] is None else f"{meta['voice_gain_db']:+}"
        print(f"  [{s['name']:<20}] 镜 {shot_start:6.2f}~{cursor:6.2f}s  "
              f"发声点+{_so}s  "
              f"字幕 {sub_start}~{sub_end}  "
              f"环境音{meta['amb_rms']}dB->{meta['amb_gain_db']:+}dB  "
              f"配音{_vr}dB->{_vg}dB", flush=True)

    write_wav(out_wav, full)
    print(f"\n混音输出: {out_wav}  {len(full)/SR:.3f}s  "
          f"peak={20*np.log10(max(float(np.abs(full).max()),1e-9)):.2f}dBFS")
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump({"total_dur": round(total, 3), "intro_dur": intro_dur,
                   "mix_wav": out_wav, "shots": rows}, f, ensure_ascii=False, indent=2)
    print(f"时间轴: {timeline_path}")
    return rows


if __name__ == "__main__":
    print("mix_audio 作为模块使用")
