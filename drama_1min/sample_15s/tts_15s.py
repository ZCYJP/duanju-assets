#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《决裂》15秒样本 — TTS 配音

按短剧语法规范 §5：
  - 人声走 Qwen3-TTS 声纹克隆（参考音色 wav），不用 RMS 归一（会反抬嘶声）
  - 每句实测时长 → atempo 轻微适配镜长（上限 1.25，避免失真）
  - 检测真实语音起止点 → 字幕时间轴由发音驱动（不是按镜长硬切）

输出：sample_15s/voice/{shot_id}.wav + voice_meta.json
"""
import os
import sys
import json
import wave
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shots_15s import SHOTS, NAME2CHAR, VOICE_REF, FPS

BASE = "/ssd2/duanju/drama_1min/sample_15s"
OUT = f"{BASE}/voice"
os.makedirs(OUT, exist_ok=True)

MODEL_DIR = "/ssd2/models/Qwen3-TTS-12Hz-1.7B-Base"
TTS_GPU = os.environ.get("TTS_GPU", "1")
SR = 24000

# 音色样本的原始文本（VoiceDesign 生成，ICL 克隆模式强制要求 ref_text）
REF_TEXT = ("一人之身，一國之象。把身体当成一个国家来看："
            "精神是君主，气是百姓，爱气才能全身。")

# 注意1：Qwen3-TTS 会把 text 原样念出，任何"语气描述"都会被读成台词
#         → 情绪只能靠音色参考本身，不能拼进 text
# 注意2：标点会造成 0.2~0.4s 的停顿，短剧节奏要紧凑
#         → TTS 文本去标点，字幕仍用原始带标点文本
TONE = {}
PUNCT = "，。、；：？！…—,.!?;: "


def strip_punct(t):
    return "".join(ch for ch in t if ch not in PUNCT)


def save_wav(path, wav):
    x = np.asarray(wav, dtype=np.float32)
    if x.ndim > 1:
        x = x.mean(axis=-1)
    mx = float(np.max(np.abs(x))) or 1.0
    x = x / mx * 0.95
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return len(pcm) / SR


def speech_bounds(x, sr=SR):
    """返回 (语音起点秒, 语音终点秒)：用 RMS 能量 + 阈值判定"""
    win = int(sr * 0.02)
    n = len(x) // win
    if n < 2:
        return 0.0, len(x) / sr
    e = np.sqrt((x[:n * win].reshape(n, win) ** 2).mean(axis=1))
    e_db = 20 * np.log10(np.maximum(e, 1e-9))
    peak = e_db.max()
    thr = max(peak - 38.0, -55.0)
    voiced = np.where(e_db > thr)[0]
    if len(voiced) == 0:
        return 0.0, len(x) / sr
    s = max(0, voiced[0] * win - win) / sr
    t = min(len(x), (voiced[-1] + 1) * win + win) / sr
    return s, t


def main():
    from qwen_tts import Qwen3TTSModel
    print("加载 TTS 模型 → cuda:%s" % TTS_GPU, flush=True)
    model = Qwen3TTSModel.from_pretrained(MODEL_DIR, device_map="cuda:%s" % TTS_GPU,
                                          dtype=torch.bfloat16)

    # 每个角色建一次声纹 prompt（省时间）
    prompts = {}
    for cid in sorted({s["dialogue"][0][0] for s in SHOTS}):
        code = NAME2CHAR[cid]
        ref, ref_name = VOICE_REF[code]
        if not os.path.exists(ref):
            print("缺失参考音色:", ref, flush=True)
            continue
        prompts[cid] = model.create_voice_clone_prompt(ref, ref_text=REF_TEXT)
        print("声纹就绪:", cid, "->", ref_name, flush=True)

    meta = []
    import librosa
    for shot in SHOTS:
        name, line = shot["dialogue"][0]
        shot_dur = shot["frames"] / FPS
        text = strip_punct(line) + TONE.get(name, "")
        torch.manual_seed(20261115 + shot["id"])
        wavs, sr = model.generate_voice_clone(text, voice_clone_prompt=prompts[name],
                                               language="Chinese", non_streaming_mode=True)
        wav = np.asarray(wavs[0], dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        raw_dur = len(wav) / sr

        # ① 裁掉 TTS 自带的首尾静音，只留语音本体
        vs, vt = speech_bounds(wav, sr)
        pad = int(0.05 * sr)
        a = max(0, int(vs * sr) - pad)
        b = min(len(wav), int(vt * sr) + pad)
        body = wav[a:b]
        speech_len = len(body) / sr

        # ② 本阶段不压缩：输出自然语速语音，镜长由外部按实测时长反推
        tempo = 1.0
        dur = len(body) / sr

        path = f"{OUT}/{shot['id']:02d}_{name}.wav"
        save_wav(path, body)
        meta.append({
            "shot_id": shot["id"], "name": name, "line": line,
            "file": os.path.basename(path), "sr": sr,
            "shot_dur": round(shot_dur, 3),
            "raw_dur": round(raw_dur, 3),
            "speech_len": round(speech_len, 3),
            "tempo": round(tempo, 3),
            "dur": round(dur, 3),
            "fill": round(dur / shot_dur, 3),
            "fit_ok": dur <= shot_dur - 0.20,
            "need_frames": int(np.ceil((dur + 0.22) / (1.0 / FPS) / 17) * 17 - 12),
        })
        print("[%d] %-4s %-12s raw=%.2f 语音本体=%.2f  当前镜长%.2fs 建议镜长≥%d帧 %s"
              % (shot["id"], name, line, raw_dur, speech_len, shot_dur,
                 meta[-1]["need_frames"],
                 "" if dur <= shot_dur - 0.20 else "!!需加长镜"), flush=True)

    json.dump(meta, open(f"{BASE}/logs/voice_meta.json", "w"), ensure_ascii=False, indent=2)
    print("\n配音完成 %d 句" % len(meta), flush=True)


if __name__ == "__main__":
    main()
