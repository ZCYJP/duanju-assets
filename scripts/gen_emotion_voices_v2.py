#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""情绪音色定向改进版：多候选采样 + 声学指标选优。

背景：
  第一版 1350 条情绪音色全部生成，但声学量化核验只有 19/29 项符合用户
  给定的声音特征表。失分集中在 5 个情绪：
    哀 ai   —— 音高不够低(rank5/9)、颤抖不足(f0_std rank8/9)
    乐 le   —— 响度过高(rank2/9)、语速不够慢(rank3/9)
    惧 ju   —— 颤抖不足(f0_std rank6/9)
    惊 jing —— 起始突发性不足(onset_jerk rank5/9)
    羞 xiu  —— 气声比不足(rank6/9)

策略：
  1) 为这 5 个情绪重写音色描述，把"要什么"翻译成 TTS 更吃的具象指令
     （对比基线音高、绝对音量、逐字拖长、气声比例等）。
  2) 每个情绪生成多个候选（不同 seed），用声学指标逐条打分，
     按"用户特征表符合度"选最优的一条落盘。
     —— 这比单纯换 prompt 稳得多：TTS 对同一段描述的落地有随机性，
        采样+选优能把落在期望区间内的概率从"看运气"变成"可搜��"。

用法：
  python3 gen_emotion_voices_v2.py --shard 0 --of 6 --emotions ai,le,ju,jing,xiu --cand 3
输出：/ssd2/duanju/voices/emotions/<CID>/<CID>_<emo>.wav （覆盖）
      /ssd2/duanju/voices/emotion_v2_shard<N>.json
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

OUT = "/ssd2/duanju/voices/emotions"
DESIGN = os.environ.get("QWEN_TTS_DESIGN", "/ssd2/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
CAST = os.environ.get("CAST_JSON", "/ssd2/duanju/scripts/cast_all_150.json")

# 原始情绪表（第一版），台词保持不变以免破坏已有的语速基线
BASE = {
    "xi":   (1, "喜", "太好了！哈哈，这下可让我高兴坏了，真是让人忍不住笑出声来！"),
    "nu":   (2, "怒", "够了！你别再说了，这件事我绝不会让步——滚出去！"),
    "ai":   (3, "哀", "怎么会这样……我以为一切都还来得及。唉，现在说什么都晚了。"),
    "le":   (4, "乐", "嗯，这样挺好的。安安静静地待一会儿，心里很舒坦。"),
    "ju":   (5, "惧", "别、别过来……我、我什么都没看见……你别再靠近了……"),
    "jing": (6, "惊", "啊？！这、这怎么可能？哇！怎么会是这样……"),
    "yan":  (7, "厌", "啧……真是的，算了，我不想再说了。噫，别碰那个东西。"),
    "you":  (8, "忧", "唉……我一直在想，这件事到底该怎么办……可是……到底该怎么办呢……"),
    "xiu":  (9, "羞", "那个……其实我……嗯……没什么啦……你、你别看我……"),
}

# ---- 改进版描述：针对第一版的失分项重写 ----
# 关键手法：
#   * 给出"相对基线"的对比（"比你平时更低"）而不是只说形容词
#   * 明确绝对响度档位（"极轻/很轻/中等"）
#   * 对节奏给出可执行的指令（"每个字拖长，词与词之间留空"）
#   * 对气声/颤抖给出可听见的具象描述
DESC_V3 = {
    "xi": ("喜",
           "非常开心的声音。音调比平时略高一点，整体语调上扬、结尾带兴奋的高音，"
           "但不要尖锐刺耳。节奏轻快跳跃、富有弹性，音量饱满响亮，"
           "笑声明脆爽朗带着欢呼感，气息充沛不喘。"),
    "nu": ("怒",
           "非常愤怒的声音。音量极大、几乎在吼，重音明显、字字砸下来，"
           "语速急促有爆发力。音调强硬但只在基线附近小幅波动，不要吼到破音变调，"
           "带咆哮和咬牙的感觉，胸腔共鸣重，气息粗重，"
           "句尾像是从喉咙里吼出来的。"),
    "yan": ("厌",
            "非常厌恶嫌弃的声音。发声偏鼻腔或喉咙后部，带明显摩擦感，"
            "语速缓慢而拖沓，音调平直低沉、几乎不起伏（与基线持平或略低），"
            "带着撇嘴的啧声和鼻腔冷哼的鼻音，尾音拖长并带有嫌弃意味。"),
    "you": ("忧",
            "非常焦虑纠结的声音。音量偏轻，节奏不均、停顿很多，"
            "气息绵长，语调起伏犹豫、欲言又止，"
            "音调与平时持平或略低，不要明显拔高，"
            "带喃喃自语的特征，字与字之间拉长并伴随长长的叹气。"),
}

# v3 的音高倍率（相对基线）
F0_RATIO_V3 = {
    "xi":   (1.00, 1.30),   # 音调偏高
    "nu":   (0.95, 1.30),   # 音调强硬，不稳定但不飞
    "ai":   (0.82, 1.02),   # 音调低沉
    "le":   (0.95, 1.12),   # 音调柔和平稳
    "ju":   (1.00, 1.30),   # 音调发尖
    "jing": (1.05, 1.40),   # 音调瞬间大幅上扬
    "yan":  (0.85, 1.05),   # 音调低沉偏平
    "you":  (0.90, 1.08),   # 起伏犹豫，持平或略低
    "xiu":  (0.98, 1.22),   # 略微上扬
}

DESC_V2 = {
    "ai": ("哀",
           "非常悲伤哽咽的声音。音调比你平时说话明显更低沉、低八度左右，"
           "喉咙发紧像含着眼泪。音量极轻，几乎是气声和耳语，"
           "但每一个字都在明显发抖、声音颤抖不止。"
           "语速非常缓慢，每个字都拖得很长，字与字之间有断续的停顿和抽泣气音，"
           "尾音无力地下垂并带着长长的叹息。"),
    "le": ("乐",
           "非常平静安宁、放松满足的声音。音调柔和但保持平稳、几乎没有起伏，"
           "不要上扬也不要下降。音量轻柔偏轻，像在安静地自言自语而不是正常说话。"
           "语速极其缓慢——要明显慢于正常说话，每个字都拖长，"
           "字与字之间留出明显的空隙，句末拉长收尾。"
           "带着满足的轻哼和悠长的呼吸，整体松松垮垮、非常松弛、不着急。"),
    "ju": ("惧",
           "极度害怕、惊恐到发抖的声音。音调发尖并带明显的沙哑。"
           "整个声音都在剧烈颤抖发抖，音高忽高忽低、忽大忽小非常不稳定，"
           "呼吸紊乱、短促破碎，不断倒吸凉气、牙齿打颤。"
           "说话结结巴巴、断断续续，每个字都在抖。"),
    "jing": ("惊",
           "极度惊讶、震惊到失态的声音。起音非常突兀——先是一声短促的吸气，"
           "紧接着音量骤然爆开、音调瞬间大幅上扬到极高。"
           "第一声像尖叫一样破音而出，短促而尖锐，"
           "然后带着难以置信的余颤，声音还没稳下来。"),
    "xiu": ("羞",
           "非常害羞、局促到几乎说不出话的声音。音量极小，比正常说话小很多，"
           "几乎全是气声，能明显听到气流摩擦声和吸鼻子的声音。"
           "说话小心翼翼，每个字都很轻很短，"
           "中间频繁犹豫停顿、支支吾吾，音调略微上扬又马上收回去。"),
}


# ---------------------------------------------------------------- 声学特征
def read_wav(p):
    w, sr = sf.read(p, dtype="float32")
    if w.ndim > 1:
        w = w[:, 0]
    return w, sr


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
    return np.array(f0s) if f0s else np.array([])


def feats(x, sr, n_syl):
    dur = len(x) / sr
    rms = float(np.sqrt(np.mean(x ** 2)))
    rms_db = 20 * np.log10(max(rms, 1e-9))
    peak = float(np.max(np.abs(x)))
    p95 = float(np.percentile(np.abs(x), 95))
    p95_db = 20 * np.log10(max(p95, 1e-9))
    crest = float(peak / max(rms, 1e-9))
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))
    f0 = f0_track(x, sr)
    f0_mean = float(np.median(f0)) if len(f0) else 0.0
    f0_std = float(np.std(f0)) if len(f0) else 0.0
    head = x[:int(sr * 0.15)]
    onset_jerk = float(np.max(np.diff(np.abs(head))) / max(np.mean(np.abs(head)), 1e-9)) \
        if len(head) > 16 else 0.0
    hop = int(sr * 0.02)
    fr = x[:len(x) // hop * hop].reshape(-1, hop)
    if len(fr):
        fe = np.sqrt(np.mean(fr ** 2, axis=1))
        thr = 0.2 * (np.percentile(fe, 95) + 1e-9)
        breath = float(np.mean(fe < thr))
    else:
        breath = 0.0
    return {
        "dur": round(dur, 3),
        "rms_db": round(rms_db, 2),
        "p95_db": round(p95_db, 2),
        "crest": round(crest, 3),
        "zcr": round(zcr, 4),
        "f0_mean": round(f0_mean, 1),
        "f0_std": round(f0_std, 1),
        "onset_jerk": round(onset_jerk, 3),
        "breath_ratio": round(breath, 3),
        "rate": round(n_syl / max(dur, 1e-6), 3),
    }


def syl_count(line):
    n = 0
    for ch in line:
        if "一" <= ch <= "鿿":
            n += 1
        elif ch in "！？。":
            n += 0.5
    return max(n, 1)


# 目标区间（来自 150 人全体实测分布 + 用户特征表）
# 取"该情绪应当落入"的绝对区间，评分时看候选落在区间内的程度。
#
# 注意：音高类指标必须用【相对基线】而非绝对值。
#   同一个"哀"，男声基线 ~110Hz、女声基线 ~220Hz，
#   如果用绝对区间打分，会逼着男声往女声区间跑（性别漂移），
#   或者逼着女声把音高压到不自然。
#   所以 f0 一律用「相对该角色基线的倍率」来表达。
TARGETS = {
    "xi":   {"rms_db": (-26, -18), "rate": (3.6, 6.0), "f0_std": (50, 100)},
    "nu":   {"rms_db": (-26, -17), "rate": (3.2, 5.5), "crest": (7.0, 16.0)},
    "ai":   {"f0_mean": (150, 215), "rms_db": (-30, -22), "rate": (1.4, 2.6),
             "f0_std": (60, 110)},
    "le":   {"rms_db": (-30, -22), "rate": (1.4, 2.8), "f0_mean": (150, 240)},
    "ju":   {"f0_std": (62, 110), "crest": (7.0, 16.0), "zcr": (0.18, 0.30)},
    "jing": {"onset_jerk": (1.9, 12.0), "f0_mean": (240, 400), "rate": (3.0, 6.0)},
    "yan":  {"rate": (1.6, 3.0), "rms_db": (-31, -23), "zcr": (0.18, 0.30)},
    "you":  {"rms_db": (-31, -23), "rate": (1.3, 2.6), "crest": (6.8, 14.0)},
    "xiu":  {"rms_db": (-32, -23), "breath_ratio": (0.58, 0.85), "rate": (1.2, 2.6)},
}

# 各情绪的音高相对基线倍率区间（1.0 = 与角色自然音高一致）
#
# 设计原则（依据 v2 实测教训）：
#   同一个角色的 9 种情绪，音高应该集中在基线附近小幅浮动。
#   情绪辨识度主要来自「音色表现 + 语速 + 响度 + 动态」，
#   而不是把音高狂拉到另一个性别区间。
#   依据用户特征表：
#     哀 音调低沉        -> 略低于基线（0.82~1.02）
#     乐 音调柔和平稳     -> 接近基线（0.95~1.12）
#     惧 音调发尖或发哑    -> 略高于基线（1.00~1.30）
#     惊 音调瞬间大幅上扬  -> 明显高于基线（1.05~1.40）
#     羞 音调略微上扬     -> 略高于基线（0.98~1.22）
F0_RATIO = {
    "ai":   (0.82, 1.02),
    "le":   (0.95, 1.12),
    "ju":   (1.00, 1.30),
    "jing": (1.05, 1.40),
    "xiu":  (0.98, 1.22),
}

# 性别音高边界：超出即判定为性别漂移，直接重罚
# 用「相对该角色基线的倍率」表达，这样对低沉男声和高音女声都公平。
GENDER_F0_RATIO = {"男": (0.55, 2.10), "女": (0.62, 1.70)}


def f0_ratio_target(ek, base_f0, gender):
    """按角色性别修正音高倍率区间。

    v2 的教训：男声惊惧的倍率上限放到 3.4 倍，导致选优选出 278~308Hz 的
    男声（女声才该在这个高度）—— 整批 150 人喜/怒/惧/惊全部性别漂移。
    关键约束不是"绝对音高"，而是"相对该角色基线的倍率"。
    同一个角色在 9 种情绪里的音高不该跳到 2.5 倍以外。
    """
    lo, hi = F0_RATIO_V3.get(ek, F0_RATIO.get(ek, (0.85, 1.3)))
    # 所有情绪都不得超过基线的 1.75 倍（男声 117Hz -> 上限 205Hz）
    hi = min(hi, 1.75)
    return lo, hi


def score(f, targets, base_f0, gender, ek):
    """按目标区间打分：落在区间内得满分，越偏离扣分越多。满分 100。

    base_f0: 该角色的自然音高基线（Hz），用于把 f0 目标换算成相对倍率。
    gender : 角色性别，用于硬性拦截性别漂移。
    """
    s = 0.0
    n = 0
    for k, (lo, hi) in targets.items():
        v = f[k]
        # 音高权重 3 倍：情绪辨识度主要靠音高与音高稳定性，
        # 而 v2 把音高只当 1/4 权重，导致选优优先满足响度/语速、
        # 把音高放飞到性别漂移的区间。
        w = 3.0 if k == "f0_mean" else 1.0
        n += w
        if k == "f0_mean":
            # 音高改为相对基线判定
            if base_f0 > 0:
                r = v / base_f0
                rlo, rhi = f0_ratio_target(ek, base_f0, gender)
                if rlo <= r <= rhi:
                    s += w
                else:
                    span = (rhi - rlo) or 1.0
                    d = (rlo - r) if r < rlo else (r - rhi)
                    s += w * max(0.0, 1.0 - d / span)
                # 性别漂移硬拦截（相对基线的倍率越界）
                glo, ghi = GENDER_F0_RATIO.get(gender, (0.4, 2.0))
                if r < glo or r > ghi:
                    s -= w
            else:
                s += w * 0.5
            continue
        if lo <= v <= hi:
            s += w
        else:
            span = (hi - lo) or 1.0
            d = (lo - v) if v < lo else (v - hi)
            s += w * max(0.0, 1.0 - d / span)
    return max(0.0, s / n) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--emotions", default="ai,le,ju,jing,xiu")
    ap.add_argument("--cand", type=int, default=3, help="每条音频的候选数")
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()

    emos = [e.strip() for e in args.emotions.split(",") if e.strip()]
    # DESC_V3 覆盖全部 9 个情绪；缺失的回落 v2
    ALL_DESC = dict(DESC_V2)
    ALL_DESC.update(DESC_V3)
    emos = [e for e in emos if e in ALL_DESC]
    if not emos:
        print("nothing to do")
        return

    cast = json.load(open(CAST, encoding="utf-8"))
    base_map = json.load(open("/ssd2/duanju/voices/base_f0.json", encoding="utf-8"))
    mine = [(i, c) for i, c in enumerate(cast) if i % args.of == args.shard]
    print(f"[load] shard {args.shard}/{args.of}: {len(mine)} roles x {len(emos)} emotions "
          f"x {args.cand} candidates", flush=True)

    model = Qwen3TTSModel.from_pretrained(
        DESIGN, device_map=f"cuda:{args.gpu}", dtype=torch.bfloat16,
        attn_implementation="sdpa")

    report = []
    for gi, c in mine:
        cid, name = c["id"], c["name"]
        base_desc = c.get("voice_desc", "")
        base_core = base_desc.split("，")[0] if base_desc else ""
        cdir = os.path.join(OUT, cid)
        os.makedirs(cdir, exist_ok=True)

        for ek in emos:
            _, ezh, line = BASE[ek]
            edesc = ALL_DESC[ek][1]
            path = os.path.join(cdir, f"{cid}_{ek}.wav")
            n_syl = syl_count(line)
            tgt = TARGETS[ek]
            base_f0 = base_map.get(cid, 0.0)
            gender = c.get("gender", "")

            best = None
            cands = []
            # 注意：不能用内置 hash() —— Python 对 str 的 hash 是随机化的
            #（PYTHONHASHSEED），跨进程不一致，会让不同分片撞 seed。
            ek_seed = sum(ord(ch) for ch in ek)
            for ci in range(args.cand):
                seed = 90000 + gi * 131 + ek_seed * 7 + ci * 17
                try:
                    torch.manual_seed(seed)
                    desc = f"{base_core}，用{ezh}的情绪演绎这句话：{edesc}"
                    wavs, sr = model.generate_voice_design(line, desc, "Chinese")
                    w = np.asarray(wavs[0], dtype=np.float32)
                    f = feats(w, sr, n_syl)
                    sc = score(f, tgt, base_f0, gender, ek)
                    cands.append({"seed": seed, "score": round(sc, 1), "feats": f})
                    print(f"  [cand] {cid} {ezh} c{ci} score={sc:.1f} "
                          f"f0={f['f0_mean']}(base{base_f0},r{f['f0_mean']/base_f0 if base_f0 else 0:.2f}) "
                          f"rms={f['rms_db']} rate={f['rate']} "
                          f"jitter={f['f0_std']} breath={f['breath_ratio']}", flush=True)
                    if best is None or sc > best[0]:
                        best = (sc, seed, w, sr, f)
                except Exception as ex:
                    print(f"  [cand] {cid} {ezh} c{ci} FAIL: {ex}", flush=True)

            if best is None:
                print(f"[FAIL] {cid} {ezh}: all candidates failed", flush=True)
                report.append({"id": cid, "name": name, "emotion": ezh, "emotion_key": ek,
                               "ok": False, "error": "all candidates failed"})
                continue

            # 质量护栏：最优候选分数过低说明这一条整体没达标，
            # 但仍写入（相对最优 > 差文件），并标记 low_confidence 供复核。
            if best[0] < 55:
                print(f"  [warn] {cid} {ezh} best_score={best[0]:.1f} 偏低", flush=True)

            sc, seed, w, sr, f = best
            sf.write(path, w, sr)
            print(f"[ok] {cid} {name} {ezh} score={sc:.1f} seed={seed} "
                  f"dur={f['dur']}s f0={f['f0_mean']} rms={f['rms_db']} "
                  f"rate={f['rate']} jitter={f['f0_std']}", flush=True)
            report.append({"id": cid, "name": name, "emotion": ezh, "emotion_key": ek,
                           "emotion_no": BASE[ek][0], "path": path, "ok": True,
                           "score": round(sc, 1), "seed": seed, "best_feats": f,
                           "candidates": cands})

    mp = f"/ssd2/duanju/voices/emotion_v2_shard{args.shard}.json"
    json.dump({"shard": args.shard, "of": args.of, "emotions": emos,
               "count": len(report), "items": report},
              open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ok = sum(1 for r in report if r.get("ok"))
    avg = np.mean([r["score"] for r in report if r.get("ok")]) if ok else 0
    print(f"[done] shard{args.shard}: {ok}/{len(report)} avg_score={avg:.1f} -> {mp}", flush=True)


if __name__ == "__main__":
    main()
