#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成片音画对齐验证 v4 —— 逐镜切段 ASR。

为什么必须逐镜切（实测演进，不是推测）：
  Whisper 会把连续的多句合并成一个大 segment。实测 s01~s06 六句被合并为
  0.74~27.66s，该 segment 内的 word_timestamps 是线性插值出来的伪精度
  （实测 s04 偏差 -1.18s、s06 +1.45s），而未被合并的 s07~s12 六句偏差
  全部在 ±0.45s 内 —— 说明偏差来自分段策略，不是音画不同步。

  所以：按成片时间轴把音频切成 12 段，每段只含一句台词，再逐段 ASR。
  这样每段的 word_timestamps 是真实对齐的。

铁律：字幕出现 == 发音起点，容差 ±0.5s。
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

TOL = 0.5
FFMPEG = "/ssd2/miniconda3/envs/videogen/bin/ffmpeg"


def char_norm(s):
    return re.sub(r"[^\u4e00-\u9fff]", "", s)


def cut_audio(video, tmpdir, segments):
    """按 [(name, start, end)] 切出 wav"""
    out = {}
    for name, st, en in segments:
        w = os.path.join(tmpdir, f"{name}.wav")
        dur = max(0.5, en - st)
        # 前后各留 1.2s 余量，避免切掉词头
        s0 = max(0.0, st - 1.2)
        subprocess.run([FFMPEG, "-y", "-v", "error", "-ss", f"{s0:.3f}",
                        "-i", video, "-t", f"{dur + 1.2:.3f}",
                        "-vn", "-ac", "1", "-ar", "16000", w], check=True)
        out[name] = (w, s0)
    return out


def main():
    video = sys.argv[1]
    tlj = sys.argv[2]
    with open(tlj, encoding="utf-8") as f:
        tl = json.load(f)
    shots = tl["shots"]

    tmpdir = tempfile.mkdtemp(prefix="align4_")
    segs = [(s["shot"], s["start"], s["end"]) for s in shots]
    wavs = cut_audio(video, tmpdir, segs)

    from faster_whisper import WhisperModel
    device = os.environ.get("ASR_DEVICE", "cpu")
    ctype = os.environ.get("ASR_COMPUTE", "int8" if device == "cpu" else "float16")
    model = WhisperModel("large-v3", device=device, compute_type=ctype,
                          download_root="/ssd2/cache/whisper")

    print("=== 逐镜 ASR（每镜只含一句台词）===")
    results = []
    for sh in shots:
        name = sh["shot"]
        wav, offset = wavs[name]
        segs_out, info = model.transcribe(
            wav, language="zh", beam_size=5, word_timestamps=True,
            vad_filter=False, condition_on_previous_text=False,
            initial_prompt="这是一段中文电视剧对白。")
        # 收集全部词（Whisper 偶尔会输出幻觉段，如「优优独播剧场」，
        # 必须按「与字幕文本最匹配」来选，而不是按「最早起点」）
        all_words = []
        seg_texts = []
        for sgm in segs_out:
            ws = [(w.start, w.end, w.word.strip()) for w in (sgm.words or [])
                  if w.word.strip()]
            if ws:
                all_words.extend(ws)
                seg_texts.append((sgm.text.strip(), ws))

        core = char_norm(sh["text"])
        # 字符级滑窗：在词序列里找与 core 最匹配的窗口
        best_score, best_start, best_text, best_words = 0.0, None, "", []
        W = max(4, len(core) + 4)
        for i in range(len(all_words)):
            j = i
            buf = []
            while j < len(all_words) and j - i < W:
                buf.append(all_words[j])
                j += 1
            t2 = char_norm("".join(w[2] for w in buf))
            if not t2:
                continue
            cover = len(set(core) & set(t2)) / max(len(set(core)), 1)
            seq = 1.0 if core in t2 else 0.0
            score = cover * 0.6 + seq * 0.4
            if score > best_score + 1e-9:
                best_score = score
                best_start = buf[0][0]
                best_text = "".join(w[2] for w in buf)
                best_words = buf

        if best_start is None:
            results.append({"shot": name, "text": sh["text"],
                            "sub_start": sh["sub_start"], "asr_start": None,
                            "delta": None, "match": "NO_ASR"})
            print(f"  [MISS] {name:<22} 未识别到语音")
            continue

        abs_start = best_start + offset
        delta = abs_start - sh["sub_start"]
        score = best_score
        results.append({"shot": name, "text": sh["text"],
                        "sub_start": sh["sub_start"], "asr_start": round(abs_start, 3),
                        "delta": round(delta, 3), "score": round(score, 3),
                        "asr_text": best_text,
                        "words": [{"s": round(a + offset, 3), "e": round(b + offset, 3),
                                   "t": t} for a, b, t in best_words]})

    deltas = [r["delta"] for r in results if r["delta"] is not None]
    med = float(np.median(deltas)) if deltas else 0.0
    for r in results:
        if r["delta"] is None:
            r["match"] = "NO_ASR"
        else:
            r["delta_norm"] = round(r["delta"] - med, 3)
            r["match"] = "OK" if abs(r["delta_norm"]) <= TOL else "OFF"

    print("\n=== 字幕 vs ASR 逐镜对齐 ===")
    for r in results:
        if r["match"] == "NO_ASR":
            print(f"  [MISS] {r['shot']:<22} 字幕{r['sub_start']:>6.2f}s  未识别")
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
        json.dump({"tolerance": TOL, "method": "per-shot segment ASR",
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"报告: {out}")
    return 0 if (off_n == 0 and miss_n == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
