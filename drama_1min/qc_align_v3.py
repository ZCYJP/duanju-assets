#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成片音画对齐验证 v3 —— 词级时间戳 + 字符级 difflib 对齐。

演进记录（都是实测踩坑，不是推测）：
  v1  只按 ASR segment 的 start 在 ±3s 内找候选
      → Whisper 把 s01~s06 合并成一个大段（0.74~27.66s），
        后续句子的字幕起点落在该段起点 3s 外 → 误判 6 个 MISS
        （实测音频本身完全正常，12 句台词都被完整识别）
  v2  改用 word_timestamps + 滑窗字符匹配 → 11 OK / 1 OFF
      → 唯一超差 s05：Whisper 把句首「爸」字划进了上一句的窗口
  v3  滑窗 + difflib 字符级对齐，把匹配起点精确映射回词索引
      → 起点取「core 首字对应的那个词」，而非「core 完整出现处」
"""
import os
import sys
import json
import subprocess
import tempfile
import re
import difflib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shots_1min import SHOTS

TOL = 0.5          # 容差（秒）
WINDOW = 8.0       # 在字幕起点 ±WINDOW 内寻找匹配
MIN_SCORE = 0.45


def char_norm(s):
    return re.sub(r"[^\u4e00-\u9fff]", "", s)


def align_start(words, core, lo, hi):
    """
    在词序列 [lo,hi) 时间范围内找与 core 最匹配的位置，
    返回 (score, 起点时间, 匹配文本)。
    起点 = core 首字所对应的那个词的 start（字符级 difflib 定位）。
    """
    if not words or not core:
        return 0.0, None, ""

    cand = [i for i, w in enumerate(words) if lo <= w["start"] <= hi]
    if not cand:
        return 0.0, None, ""

    W = max(4, len(core) + 4)
    best = (0.0, None, "", None, None)

    for i in cand:
        j = i
        buf = []
        while j < len(words) and words[j]["start"] <= hi and j - i < W:
            buf.append(words[j]["text"])
            j += 1
        t2 = char_norm("".join(buf))
        if not t2:
            continue
        cover = len(set(core) & set(t2)) / max(len(set(core)), 1)
        seq = 1.0 if core in t2 else 0.0
        score = cover * 0.6 + seq * 0.4
        if score <= best[0] + 1e-9:
            continue

        # 字符级定位：core 在窗口串里的最佳匹配块起点
        sm = difflib.SequenceMatcher(None, core, t2)
        blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
        if blocks:
            # 取匹配 core 首字的块（block.a == 0 最佳；否则取最靠前块）
            first = blocks[0]
            char_off = first.b
        else:
            char_off = 0
        # 把字符偏移映射回词索引
        acc_len = 0
        target_word = i
        for k in range(i, j):
            clen = len(char_norm(words[k]["text"]))
            if acc_len + clen > char_off:
                target_word = k
                break
            acc_len += clen
        best = (score, words[target_word]["start"], "".join(buf), i, j)

    return best[0], best[1], best[2]


def main():
    video = sys.argv[1]
    tlj = sys.argv[2]
    with open(tlj, encoding="utf-8") as f:
        tl = json.load(f)
    shots = tl["shots"]

    tmpdir = tempfile.mkdtemp(prefix="align3_")
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
    asr, words = [], []
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
        lo = max(0.0, sub_start - 2.0)
        hi = sub_start + WINDOW
        score, t0, mt = align_start(words, core, lo, hi)
        if t0 is None or score < MIN_SCORE:
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
        json.dump({"tolerance": TOL, "method": "word-level + difflib char alignment",
                   "asr_segments": asr, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"报告: {out}")
    return 0 if (off_n == 0 and miss_n == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
