#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证：把角色 voice_desc 的音高约束信息带入情绪 prompt，能否压住音高漂移。

背景
----
gen_emotion_voices_v2.py 里：
    base_core = base_desc.split("，")[0]     # 只取第一段
对 M01 陆沉，voice_desc = "三十岁左右成年男性的声音，上市公司总裁，标准普通话，
声音低沉浑厚富有磁性，咬字有力字字清晰，语速沉稳从容，带不容置疑的上位者威压，
不夸张不咆哮"
base_core 只拿到 "三十岁左右成年男性的声音" ——
"声音低沉浑厚" "不夸张不咆哮" 这些直接约束音高与音量的信息全被丢掉了。

于是当情绪描述要求"喜/怒/惊"的高唤醒时，
模型失去了唯一的音高锚点 → 男声被拉到 260Hz+（女声音域）。

本探针对同一角色（男声 + 女声）对比三种 prompt 构造：
  A. base_core（现状，只有第一段）
  B. base_core + 音色关键段（保留"低沉/清润/明亮"等音高锚点）
  C. B + 显式音高锚定（直接写明"音高保持在平时的基线水平"）
在每个情绪上各生成 4 个候选，看 f0 倍率分布。
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/ssd2/ttslib")
from qwen_tts import Qwen3TTSModel  # noqa: E402

DESIGN = "/ssd2/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
GPU = sys.argv[1] if len(sys.argv) > 1 else "0"
CAND = 3

BASE = {
    "xi": ("喜", "太好了！哈哈，这下可让我高兴坏了，真是让人忍不住笑出声来！"),
    "nu": ("怒", "够了！你别再说了，这件事我绝不会让步——滚出去！"),
    "ju": ("惧", "别、别过来……我、我什么都没看见……你别再靠近了……"),
    "jing": ("惊", "啊？！这、这怎么可能？哇！怎么会是这样……"),
}
# 情绪描述（沿用 v3 的重写版）
EDESC = {
    "xi": ("喜", "非常开心的声音。音调比平时略高一点，整体语调上扬、结尾带兴奋的高音，"
                 "但不要尖锐刺耳。节奏轻快跳跃、富有弹性，音量饱满响亮，"
                 "笑声明脆爽朗带着欢呼感，气息充沛不喘。"),
    "nu": ("怒", "非常愤怒的声音。音量极大、几乎在吼，重音明显、字字砸下来，"
                 "语速急促有爆发力。音调强硬但只在基线附近小幅波动，不要吼到破音变调，"
                 "带咆哮和咬牙的感觉，胸腔共鸣重，气息粗重，"
                 "句尾像是从喉咙里吼出来的。"),
    "ju": ("惧", "极度害怕、惊恐到发抖的声音。音调发尖并带明显的沙哑。"
                 "整个声音都在剧烈颤抖发抖，音高忽高忽低、忽大忽小非常不稳定，"
                 "呼吸紊乱、短促破碎，不断倒吸凉气、牙齿打颤。"
                 "说话结结巴巴、断断续续，每个字都在抖。"),
    "jing": ("惊", "极度惊讶、震惊到失态的声音。起音非常突兀——先是一声短促的吸气，"
                   "紧接着音量骤然爆开、音调瞬间大幅上扬到极高。"
                   "第一声像尖叫一样破音而出，短促而尖锐，"
                   "然后带着难以置信的余颤，声音还没稳下来。"),
}

# 音高/音色锚点关键词：命中则保留该段
ANCHOR_KW = ["低沉", "浑厚", "清润", "温厚", "明亮", "清亮", "偏高", "偏低",
             "沙哑", "磁性", "清朗", "柔美", "嗓音", "声线", "音域", "不夸张",
             "不咆哮", "克制", "平稳", "沉稳"]


def pick_anchor(desc):
    """从 voice_desc 中挑出携带音高/音色约束的片段。"""
    segs = [s.strip() for s in desc.split("，") if s.strip()]
    hit = [s for s in segs[1:] if any(k in s for k in ANCHOR_KW)]
    return hit


def f0_track(x, sr):
    hop = int(sr * 0.04)
    win = int(sr * 0.04)
    lo, hi = int(sr / 400), int(sr / 70)
    f0s = []
    for i in range(0, len(x) - win, hop):
        fr = x[i:i + win]
        if np.sqrt(np.mean(fr ** 2)) < 0.005:
            continue
        fr = fr - fr.mean()
        c = np.correlate(fr, fr, "full")[win - 1:]
        if len(c) <= hi:
            continue
        seg = c[lo:hi]
        if len(seg) == 0:
            continue
        k = int(np.argmax(seg)) + lo
        if c[k] <= 0.3 * c[0] or c[k] <= 0:
            continue
        f0s.append(sr / k)
    return np.array(f0s) if len(f0s) else np.array([])


def med_f0(x, sr):
    f = f0_track(x, sr)
    return float(np.median(f)) if len(f) else 0.0


def base_f0_of(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x[:, 0]
    return med_f0(x, sr)


cast = json.load(open("/ssd2/duanju/scripts/cast_all_150.json", encoding="utf-8"))
base_map = json.load(open("/ssd2/duanju/voices/base_f0.json", encoding="utf-8"))

picks = []
for c in cast:
    if c["gender"] == "男" and not any(p[0]["gender"] == "男" for p in picks):
        picks.append((c, "男"))
    if c["gender"] == "女" and not any(p[0]["gender"] == "女" for p in picks):
        picks.append((c, "女"))
    if len(picks) == 2:
        break

model = Qwen3TTSModel.from_pretrained(
    DESIGN, device_map=f"cuda:{GPU}", dtype=torch.bfloat16,
    attn_implementation="sdpa")
print(f"[load] ok cuda:{GPU}", flush=True)

summary = {}
for c, g in picks:
    cid = c["id"]
    desc = c.get("voice_desc", "")
    base_f0 = base_map.get(cid, 0.0)
    first = desc.split("，")[0]
    anchor = pick_anchor(desc)
    anchor_txt = "，".join(anchor[:3]) if anchor else ""

    print(f"\n===== {cid} {c['name']} ({g}) base_f0={base_f0} =====", flush=True)
    print(f"  [现状 base_core] {first}", flush=True)
    print(f"  [锚点片段] {anchor_txt or '(无)'}",
          flush=True)

    variants = {
        "A_base_core": first,
        "B_anchor": (first + "，" + anchor_txt) if anchor_txt else first,
        "C_anchor_lock": ((first + "，" + anchor_txt + "，")
                          if anchor_txt else first + "，")
                         + "音高保持与你平时的自然基线一致，不要因为情绪变化而改变音域",
    }

    for ek, (ezh, line) in BASE.items():
        for vname, prefix in variants.items():
            rs = []
            for ci in range(CAND):
                try:
                    torch.manual_seed(7000 + ci * 31 + sum(ord(ch) for ch in ek))
                    full = f"{prefix}，用{ezh}的情绪演绎这句话：{EDESC[ek][1]}"
                    wavs, sr = model.generate_voice_design(line, full, "Chinese")
                    w = np.asarray(wavs[0], dtype="float32")
                    m = med_f0(w, sr)
                    r = m / base_f0 if base_f0 else 0
                    rs.append(r)
                except Exception as ex:
                    print(f"    [{ek}/{vname}] FAIL {type(ex).__name__}: {ex}",
                          flush=True)
            if rs:
                mr = float(np.mean(rs))
                print(f"  [{ezh}] {vname:14s} r: " +
                      " ".join(f"{x:.2f}" for x in rs) +
                      f"  mean={mr:.2f}", flush=True)
                summary.setdefault((g, ek, vname), []).append(mr)

print("\n===== 汇总（男女各取1人，仅示意）=====")
for k, v in summary.items():
    print(f"  {k}: mean_r={np.mean(v):.2f}")
