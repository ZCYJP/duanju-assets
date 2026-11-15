#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成片音画对齐验证 v2 —— 词级时间戳匹配。

v1 的缺陷：只按 ASR segment 的 start 在 ±3s 内找候选。当 Whisper 把多句合并成
一个长 segment（实测 s01~s06 被合并为 0.74~27.66s）时，后续句子的字幕起点
落在该 segment 起点 3s 之外 → 全部误判为 MISS。
（实测中音频完全正常，12 句台词都被完整识别，只是分段粒度不同。）

v2 方案：用 word_timestamps 拿到每个词的精确起止，再在词序列上做滑窗字符匹配，
定位每句台词的第一个词，从而得到真实发声点。
"""
import os
import sys
import json
import subprocess
import tempfile
import re

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shots_1min import SHOTS

TOL = 0.5          # 容差（秒）
WINDOW = 8.0       # 在字幕起点 ±WINDOW 内寻找匹配


def char_norm(s):
    return re.sub(r"[^\u4e00-\u9fff]", "", s)


def best_window(words, core, lo, hi):
    """
    在词序列的 [lo, hi) 时间范围内滑窗，寻找与 core 字符集重合度最高的窗口。
    返回 (score, start_time, matched_text)
    """
    if not words or not core:
        return 0.0, None, ""
    core_set = set(core)
    n = len(core_set)
    best_score = 0.0
    best_txt = ""
    best_i = best_j = None
    # 窗口比 core 长 4 个词：Whisper 常漏识句首字（实测 s05 漏「爸」），
    # 窗口需能容纳前后各若干词才能取到真实发声点
    W = max(4, len(core) + 4)
    for i in range(len(words)):
        if words[i]["start"] < lo or words[i]["start"] > hi:
            continue
        j = i
        buf = []
        while j < len(words) and words[j]["start"] <= hi:
            buf.append(words[j]["text"])
            if j - i + 1 >= W:
                break
            j += 1
        t2 = char_norm("".join(buf))
        if not t2:
            continue
        # 字符集覆盖 + 序列包含 双指标
        cover = len(core_set & set(t2)) / n
        seq = 1.0 if core in t2 else 0.0
        score = cover * 0.6 + seq * 0.4
        if score > best_score + 1e-9:
            best_score = score
            best_txt = "".join(buf)
            best_i, best_j = i, j

    if best_i is None:
        return 0.0, None, ""

    # 精修起点：窗口可能含前一句的词导致起点偏早，
    # 在最佳窗口内定位 core 首次出现处对应的词，取其真实起点
    start_t = words[best_i]["start"]
    acc = ""
    for k in range(best_i, best_j):
        acc += words[k]["text"]
        if core in char_norm(acc):
            start_t = words[k]["start"]
            break
    return best_score, start_t, best_txt


def main():
    video = sys.argv[1]
    tlj = sys.argv[2]
    with open(tlj, encoding="utf-8") as f:
        tl = json.load(f)
    shots = tl["shots"]

    tmpdir = tempfile.mkdtemp(prefix="align2_")
    wav = os.path.join(tmpdir, "a.wav")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", video,
                    "-vn", "-ac", "1", "-ar", "16000", wav], check=True)

    from faster_whisper import WhisperModel
    device = os.environ.get("ASR_DEVICE", "cpu")
    ctype = os.environ.get("ASR_COMPUTE", "int8" if device == "cpu" else "float16")
    model = WhisperModel("large-v3", device=device, compute_type=ctype,
                          download_root="/ssd2/cache/whisper")
    segs, info = model.transcribe(wav, language="zh", beam_size=5,
                                  word_timestamps=True, vad_filter=False,
                                  condition_on_previous_text=False)
    asr = []
    words = []
    for s in segs:
        asr.append({"start": round(s.start, 3), "end": round(s.end, 3),
                    "text": s.text.strip()})
        for w in (s.words or []):
            t = w.word.strip()
            if t:
                words.append({"start": round(w.start, 3), "end": round(w.end, 3),
                              "text": t})
    print("=== ASR 分句 ===")
    for a in asr:
        print(f"  {a['start']:>7.2f}~{a['end']:>7.2f}  {a['text']}")
    print(f"\n=== ASR 词数: {len(words)} ===")

    print("\n=== 字幕 vs ASR 词级对齐 ===")
    results = []
    for sh in shots:
        text = sh["text"]
        core = char_norm(text)
        sub_start = sh["sub_start"]
        lo = max(0.0, sub_start - 1.5)
        hi = sub_start + WINDOW
        score, t0, mt = best_window(words, core, lo, hi)
        if t0 is None or score < 0.45:
            results.append({"shot": sh["shot"], "text": text, "sub_start": sub_start,
                            "asr_start": None, "delta": None, "score": round(score, 3),
                            "match": "NO_ASR"})
            print(f"  [MISS] {sh['shot']:<22} 字幕{sub_start:>6.2f}s  未匹配  "
                  f"\"{text}\"  (score={score:.2f})")
            continue
        delta = t0 - sub_start
        results.append({"shot": sh["shot"], "text": text, "sub_start": sub_start,
                        "asr_start": round(t0, 3), "delta": round(delta, 3),
                        "score": round(score, 3), "asr_text": mt[:60]})

    deltas = [r["delta"] for r in results if r["delta"] is not None]
    med = float(np.median(deltas)) if deltas else 0.0
    for r in results:
        if r["delta"] is None:
            r["match"] = "NO_ASR"
        else:
            r["delta_norm"] = round(r["delta"] - med, 3)
            r["match"] = "OK" if abs(r["delta_norm"]) <= TOL else "OFF"
    print()
    for r in results:
        if r["match"] == "NO_ASR":
            print(f"  [MISS] {r['shot']:<22} 字幕{r['sub_start']:>6.2f}s  未匹配  \"{r['text']}\"")
        else:
            flag = "OK  " if r["match"] == "OK" else "OFF "
            print(f"  [{flag}] {r['shot']:<22} 字幕{r['sub_start']:>6.2f}s  "
                  f"ASR{r['asr_start']:>6.2f}s  Δ={r['delta']:+.2f}s  "
                  f"Δ归一={r['delta_norm']:+.2f}s  score={r['score']:.2f}  "
                  f"\"{r['text']}\"  -> {r['asr_text']}")

    ok_n = sum(1 for r in results if r["match"] == "OK")
    off_n = sum(1 for r in results if r["match"] == "OFF")
    miss_n = sum(1 for r in results if r["match"] == "NO_ASR")
    print(f"\n对齐: OK {ok_n} / 偏移 {off_n} / 未匹配 {miss_n} / 共 {len(results)}")
    print(f"容差 ±{TOL}s，ASR 系统性偏移中位数 {med:+.3f}s（已归一化）")
    mx = max((abs(r["delta_norm"]) for r in results if r["delta"] is not None), default=0)
    print(f"归一化后最大绝对偏差: {mx:.3f}s")

    out = os.path.join(os.path.dirname(tlj), "align_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"tolerance": TOL, "method": "word-level sliding window",
                   "asr_segments": asr, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"报告: {out}")
    return 0 if (off_n == 0 and miss_n == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
