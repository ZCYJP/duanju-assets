#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成片音画对齐验证 — 用 faster-whisper 对成片做带时间戳的 ASR，
逐句比对「ASR 识别到的语音起点」与「ASS 字幕起点」。

铁律：字幕出现 == 发音起点，容差 0.5s。

用法：qc_align.py <成片mp4> <timeline.json>
"""
import os, sys, json, subprocess, tempfile, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shots_1min import SHOTS

TOL = 0.5   # 容差（秒）


def main():
    video = sys.argv[1]
    tlj = sys.argv[2]
    with open(tlj, encoding="utf-8") as f:
        tl = json.load(f)
    shots = tl["shots"]

    # 提取音频到 wav
    tmpdir = tempfile.mkdtemp(prefix="align_")
    wav = os.path.join(tmpdir, "a.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video,
                    "-vn", "-ac", "1", "-ar", "16000", wav], check=True)

    from faster_whisper import WhisperModel
    # 注：本机 venv_tts 的 cublas 与 torch cu130 不匹配，GPU 推理会报
    # "Library libcublas.so.12 is not found"，改用 CPU int8 推理（60s 音频可接受）
    device = os.environ.get("ASR_DEVICE", "cpu")
    ctype = os.environ.get("ASR_COMPUTE", "int8" if device == "cpu" else "float16")
    model = WhisperModel("large-v3", device=device, compute_type=ctype,
                          download_root="/ssd2/cache/whisper")
    segs, info = model.transcribe(wav, language="zh", beam_size=5,
                                  word_timestamps=True, vad_filter=False)
    asr = []
    for s in segs:
        asr.append({"start": round(s.start, 3), "end": round(s.end, 3),
                    "text": s.text.strip()})
    print("=== ASR 分句 ===")
    for a in asr:
        print(f"  {a['start']:>7.2f}~{a['end']:>7.2f}  {a['text']}")

    # 逐句匹配：在字幕起点 ±WINDOW 内，选字符重合度最高的 ASR 段
    # （不能用「首个关键词命中」——同音字/共用字会导致跨句误配）
    print("\n=== 字幕 vs ASR 对齐 ===")
    WINDOW = 3.0
    results = []
    for sh in shots:
        text = sh["text"]
        core = re.sub(r"[^\u4e00-\u9fff]", "", text)
        cands = [a for a in asr
                 if abs(a["start"] - sh["start"]) <= WINDOW]
        best, best_score = None, -1
        for a in cands:
            t2 = re.sub(r"[^\u4e00-\u9fff]", "", a["text"])
            if not t2 or not core:
                continue
            inter = len(set(core) & set(t2))
            score = inter / max(len(set(core)), 1)
            if score > best_score:
                best, best_score = a, score
        if best is None or best_score < 0.4:
            results.append({"shot": sh["shot"], "text": text, "sub_start": sh["start"],
                            "asr_start": None, "delta": None, "match": "NO_ASR"})
            print(f"  [MISS] {sh['shot']:<22} 字幕{sh['start']:>6.2f}s  未匹配到 ASR  \"{text}\"")
            continue
        delta = best["start"] - sh["start"]
        results.append({"shot": sh["shot"], "text": text, "sub_start": sh["start"],
                        "asr_start": best["start"], "delta": round(delta, 3),
                        "score": round(best_score, 3),
                        "asr_text": best["text"]})

    # whisper 的 segment 边界系统性偏晚，取中位数偏移做归一化后再判定
    deltas = [r["delta"] for r in results if r["delta"] is not None]
    med = float(np.median(deltas)) if deltas else 0.0
    for r in results:
        if r["delta"] is None:
            r["match"] = "NO_ASR"
        else:
            r["delta_norm"] = round(r["delta"] - med, 3)
            r["match"] = "OK" if abs(r["delta_norm"]) <= TOL else "OFF"
    for r in results:
        if r["match"] == "NO_ASR":
            print(f"  [MISS] {r['shot']:<22} 字幕{r['sub_start']:>6.2f}s  未匹配  \"{r['text']}\"")
        else:
            print(f"  [{'OK  ' if r['match']=='OK' else 'OFF '}] {r['shot']:<22} "
                  f"字幕{r['sub_start']:>6.2f}s  ASR{r['asr_start']:>6.2f}s  "
                  f"Δ={r['delta']:+.2f}s  Δ归一={r['delta_norm']:+.2f}s  "
                  f"\"{r['text']}\"  -> {r['asr_text']}")

    ok_n = sum(1 for r in results if r["match"] == "OK")
    off_n = sum(1 for r in results if r["match"] == "OFF")
    miss_n = sum(1 for r in results if r["match"] == "NO_ASR")
    print(f"\n对齐: OK {ok_n} / 偏移 {off_n} / 未匹配 {miss_n} / 共 {len(results)}")
    print(f"容差 ±{TOL}s，ASR 系统性偏移中位数 {med:+.3f}s（已归一化）")
    print(f"归一化后最大绝对偏差: {max((abs(r['delta_norm']) for r in results if r['delta'] is not None), default=0):.3f}s")

    with open("/ssd2/duanju/drama_1min/final/align_report.json", "w", encoding="utf-8") as f:
        json.dump({"tolerance": TOL, "asr_segments": asr, "results": results},
                  f, ensure_ascii=False, indent=2)
    print("报告: /ssd2/duanju/drama_1min/final/align_report.json")
    return 0 if (off_n == 0 and miss_n == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
