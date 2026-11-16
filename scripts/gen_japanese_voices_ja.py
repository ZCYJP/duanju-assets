#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""日本角色 · 日语音色（VoiceDesign language="Japanese"）。

与中文版 gen_japanese_voices.py 并行产出，互不覆盖：
  中文版 → /ssd2/duanju/voices/japanese/<ID>/<ID>_<emo>.wav
  日文版 → /ssd2/duanju/voices/japanese_ja/<ID>/<ID>_<emo>.wav

用法：
  python3 gen_japanese_voices_ja.py --shard 0 --of 4 --gpu 0 --cand 3
  python3 gen_japanese_voices_ja.py --only F01,M03,M22 --cand 8 --gpu 0
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

sys.path.insert(0, '/ssd2/duanju/scripts')
import gen_emotion_voices_v2 as V2

OUT = "/ssd2/duanju/voices/japanese_ja"
DESIGN = os.environ.get("QWEN_TTS_DESIGN", "/ssd2/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
CAST = os.environ.get("CAST_JSON", "/ssd2/duanju/scripts/japanese_100.json")
BASE_F0_DIR = "/ssd2/duanju/voices/base_f0_japanese_ja"
MATCH_REPORT = "/ssd2/duanju/voices/ja_match_report.json"

BASE_LINE = "今日はいい天気ですね。近くを少し歩いてみます。"

# 9 情绪的日文台词 + 日文演绎指示
JA = {
    "xi":   ("喜", "本当によかった！この知らせ、ずっと待っていました。",
             "明るい笑顔、喜びに満ちた声、語尾が弾む"),
    "nu":   ("怒", "もういい！こんな無礼な行為、もう許せません！",
             "怒り、強い口調、声が大きくなる、語調が鋭い"),
    "ai":   ("哀", "なぜこうなったの…。私は必死に努力したのに。",
             "悲しみ、低い声、落胆した調子、息が重い"),
    "le":   ("楽", "今日はいい天気ですね。いっしょに散歩しましょう。",
             "楽しそう、穏やかな声、柔らかい調子"),
    "ju":   ("惧", "あれは…何ですか？近づかないでください！",
             "恐怖、震える声、息が乱れる、怯えている"),
    "jing": ("驚", "えっ！？どうしてここにいるんですか！？",
             "驚き、声が大きく上擦る、息を呑む"),
    "yan":  ("厌", "もう聞きたくないです。やめてください。",
             "退屈、いらいらした声、語気が冷たく dryness を含む"),
    "you":  ("憂", "ずっと心配で、胸が落ち着きません。",
             "心配、落ち着かない声、溜息が混じる"),
    "xiu":  ("羞", "わ、わっ…私、まだ準備ができていません…",
             "恥ずかしさ、声が小さくなる、はにかむ"),
}

# 角色日文描述：职业 + 音色特征全部映射为日文（japanese_ja_dict.py）
try:
    from japanese_ja_dict import ROLE_JA, FEAT_JA, LOOK_JA
except ImportError:
    sys.path.insert(0, '/ssd2/duanju/scripts')
    from japanese_ja_dict import ROLE_JA, FEAT_JA, LOOK_JA

GENDER_JA = {"男": "男性", "女": "女性"}

# 日文专用目标区间（按日文实测分布校准）。
#
# 语速口径差异：中文版 rate 用「汉字数/秒」，日文版用「假名数/秒」，
# 同一句话的日文假名数约为中文汉字数的 1.4 倍 → rate 系统性偏高约 1.4 倍。
# 直接套用中文区间会把「哀/乐/忧/厌」整批判为语速不达标（实测 le avg 47.8）。
# 故慢速类情绪的 rate 区间整体上调，响度与音高抖动按日文实测分布放宽。
TARGETS_JA = {
    "xi":   {"rms_db": (-26, -18), "rate": (3.6, 6.0), "f0_std": (50, 100)},
    "nu":   {"rms_db": (-26, -17), "rate": (3.2, 5.5), "crest": (7.0, 16.0)},
    "ai":   {"f0_mean": (150, 215), "rms_db": (-30, -20), "rate": (2.8, 4.6),
             "f0_std": (45, 110)},
    "le":   {"rms_db": (-28, -18), "rate": (3.0, 5.8), "f0_mean": (150, 240)},
    "ju":   {"f0_std": (62, 110), "crest": (7.0, 16.0), "zcr": (0.18, 0.30)},
    "jing": {"onset_jerk": (1.5, 12.0), "f0_mean": (240, 400), "rate": (2.6, 6.0)},
    "yan":  {"rate": (2.2, 4.2), "rms_db": (-31, -21), "zcr": (0.18, 0.30)},
    "you":  {"rms_db": (-31, -21), "rate": (1.9, 3.7), "crest": (6.8, 14.0)},
    "xiu":  {"rms_db": (-32, -21), "breath_ratio": (0.58, 0.85), "rate": (1.7, 3.7)},
}

# 日文音高倍率区间（相对角色基线）。
# 中文版区间对日文太窄：日文语调与中文不同，同一情绪的音高偏移量不同。
# 实测日文「乐」ratio p25=0.96 / med=1.09 / p75=1.18（中文目标 0.95~1.12 只覆盖约一半），
# 「哀」p25=0.86 / med=0.94 / p75=1.04（中文目标 0.82~1.02 覆盖约七成）。
# 按日文实测分布放宽，仍保持「同一角色 9 种情绪音高集中在基线附近」的设计意图。
F0_RATIO_JA = {
    "ai":   (0.78, 1.12),
    "le":   (0.88, 1.25),
    "ju":   (0.95, 1.35),
    "jing": (1.05, 2.60),
    "xiu":  (0.95, 1.28),
}


def score_ja(f, tgt, base_f0, gender, ek):
    """按日文倍率区间打分（临时覆盖 V2 的全局倍率表）。"""
    saved = (V2.F0_RATIO_V3.get(ek), V2.F0_RATIO.get(ek))
    if ek in F0_RATIO_JA:
        V2.F0_RATIO_V3[ek] = F0_RATIO_JA[ek]
        V2.F0_RATIO[ek] = F0_RATIO_JA[ek]
    try:
        return V2.score(f, tgt, base_f0, gender, ek)
    finally:
        if saved[0] is not None:
            V2.F0_RATIO_V3[ek] = saved[0]
        if saved[1] is not None:
            V2.F0_RATIO[ek] = saved[1]

# 「惊」用绝对音高判性别漂移（相对倍率对剧烈上扬情绪天然失真）
JING_F0_RATIO = (1.05, 2.60)
GENDER_ABS_F0 = {"男": (70.0, 320.0), "女": (110.0, 430.0)}


def score_jing(f, base_f0, gender):
    """惊情绪专用打分：音高按放宽倍率 + 绝对音高性别边界判定。"""
    s = 0.0
    n = 0
    for k, (lo, hi) in TARGETS_JA["jing"].items():
        v = f[k]
        w = 3.0 if k == "f0_mean" else 1.0
        n += w
        if k == "f0_mean":
            if base_f0 > 0:
                r = v / base_f0
                if JING_F0_RATIO[0] <= r <= JING_F0_RATIO[1]:
                    s += w
                else:
                    span = (JING_F0_RATIO[1] - JING_F0_RATIO[0]) or 1.0
                    d = (JING_F0_RATIO[0] - r) if r < JING_F0_RATIO[0] else (r - JING_F0_RATIO[1])
                    s += w * max(0.0, 1.0 - d / span)
            else:
                s += w * 0.5
            glo, ghi = GENDER_ABS_F0.get(gender, (60.0, 450.0))
            if v < glo or v > ghi:
                s -= w
            continue
        if lo <= v <= hi:
            s += w
        else:
            span = (hi - lo) or 1.0
            d = (lo - v) if v < lo else (v - hi)
            s += w * max(0.0, 1.0 - d / span)
    return max(0.0, s / n) * 100.0


def ja_syl_count(line):
    """日语音节数：假名各计 1，汉字计 1，标点计 0.5。"""
    n = 0.0
    for ch in line:
        o = ord(ch)
        if 0x3041 <= o <= 0x309F or 0x30A0 <= o <= 0x30FF:
            n += 1
        elif 0x4E00 <= o <= 0x9FFF:
            n += 1
        elif ch in "！？。、…":
            n += 0.5
    return max(n, 1)


def load_age_override():
    """以形象实际年龄为准修正音色描述的年龄。

    VL 看 hero 图判出的年龄与配置年龄差 >8 岁时，说明形象显老/显年轻，
    音色若仍按配置年龄生成就会与形象对不上 → 用形象年龄覆盖。
    """
    ov = {}
    if not os.path.exists(MATCH_REPORT):
        return ov
    try:
        rep = json.load(open(MATCH_REPORT, encoding="utf-8"))
    except Exception:
        return ov
    for r in rep.get("rows", []):
        if not r.get("ok"):
            continue
        ai, ac = r.get("age_img", 0), r.get("age_cfg", 0)
        if ai and ac and abs(ai - ac) > 8:
            ov[r["id"]] = ai
    return ov


AGE_OVERRIDE = load_age_override()


def role_desc(c):
    """组装纯日文音色描述。

    与人物形象同源：age / gender / role / look / voice_desc 全部来自
    japanese_100.json（形象 desc 与音色描述共用同一份配置），
    再把 look 的外貌气质映射成日文印象词，保证音色与形象气质一致。
    年龄以形象实际年龄为准（见 load_age_override）。
    """
    cid = c["id"]
    age = AGE_OVERRIDE.get(cid, c.get("age", 35))
    g = GENDER_JA.get(c.get("gender", ""), "人物")
    role = ROLE_JA.get(c.get("role", ""), "")
    look = LOOK_JA.get(c.get("look", ""), "")
    parts = [f"{age}歳の日本{g}"]
    if role:
        parts.append(role)
    if look:
        parts.append(look)
    feats = []
    for p in c.get("voice_desc", "").split("，")[3:]:
        ja = FEAT_JA.get(p)
        if ja:
            feats.append(ja)
    if feats:
        parts.append("、".join(feats))
    return "、".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--cand", type=int, default=3)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--only", default=None)
    ap.add_argument("--emotions", default=None)
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args()

    cast = json.load(open(CAST, encoding="utf-8"))
    if args.only:
        keep = set(x.strip() for x in args.only.split(',') if x.strip())
        cast = [c for c in cast if c['id'] in keep]
    mine = [(i, c) for i, c in enumerate(cast) if i % args.of == args.shard]
    emos = [e.strip() for e in (args.emotions or ",".join(JA.keys())).split(',') if e.strip()]
    emos = [e for e in emos if e in JA]
    print(f"[load] shard {args.shard}/{args.of}: {len(mine)} roles x {len(emos)} emotions "
          f"x {args.cand} candidates", flush=True)

    os.makedirs(BASE_F0_DIR, exist_ok=True)
    base_map = {}
    for bf in sorted(os.listdir(BASE_F0_DIR)):
        if bf.endswith(".json"):
            try:
                base_map.update(json.load(open(os.path.join(BASE_F0_DIR, bf), encoding="utf-8")))
            except Exception:
                pass

    model = Qwen3TTSModel.from_pretrained(
        DESIGN, device_map=f"cuda:{args.gpu}", dtype=torch.bfloat16,
        attn_implementation="sdpa")

    report = []
    for gi, c in mine:
        cid = c["id"]
        gender = c.get("gender", "")
        rdesc = role_desc(c)
        cdir = os.path.join(OUT, cid)
        os.makedirs(cdir, exist_ok=True)

        base_f0 = base_map.get(cid, 0.0)
        if base_f0 <= 0:
            try:
                torch.manual_seed(940000 + gi * 131)
                wavs, sr = model.generate_voice_design(BASE_LINE, rdesc, "Japanese")
                w = np.asarray(wavs[0], dtype=np.float32)
                bf = V2.feats(w, sr, ja_syl_count(BASE_LINE))
                base_f0 = bf["f0_mean"]
                base_map[cid] = base_f0
                # 增量写盘：基线音高是后续 f0 相对判定与性别漂移检查的唯一依据，
                # 若只在脚本末尾统一写，中途崩溃就会全部丢失。
                bf_path = os.path.join(BASE_F0_DIR,
                                       f"base_shard{args.shard}_of{args.of}.json")
                json.dump(base_map, open(bf_path, "w", encoding="utf-8"),
                          ensure_ascii=False, indent=1)
                print(f"  [base] {cid} f0={base_f0}", flush=True)
            except Exception as ex:
                print(f"  [base] {cid} FAIL: {ex}", flush=True)
                base_f0 = 0.0

        for ek in emos:
            ezh, line, edesc = JA[ek]
            path = os.path.join(cdir, f"{cid}_{ek}.wav")
            if os.path.exists(path) and os.path.getsize(path) > 1024 and not args.regen:
                print(f"  [skip] {cid} {ezh} exists", flush=True)
                continue
            n_syl = ja_syl_count(line)
            tgt = TARGETS_JA[ek]
            best = None
            cands = []
            ek_seed = sum(ord(ch) for ch in ek)
            for ci in range(args.cand):
                seed = 940000 + gi * 131 + ek_seed * 7 + ci * 17
                try:
                    torch.manual_seed(seed)
                    desc = f"{rdesc}。{ezh}の感情で表現する：{edesc}"
                    wavs, sr = model.generate_voice_design(line, desc, "Japanese")
                    w = np.asarray(wavs[0], dtype=np.float32)
                    f = V2.feats(w, sr, n_syl)
                    sc = score_jing(f, base_f0, gender) if ek == "jing" \
                        else score_ja(f, tgt, base_f0, gender, ek)
                    cands.append({"seed": seed, "score": round(sc, 1), "feats": f})
                    print(f"  [cand] {cid} {ezh} c{ci} score={sc:.1f} f0={f['f0_mean']} "
                          f"rms={f['rms_db']} rate={f['rate']} dur={f['dur']}", flush=True)
                    # 时长失控的候选（>12s）直接排除：日文长句偶发拖尾，
                    # 会导致尾部幻觉填充（实测 F01_ai 首版 15.3s 尾部乱码）
                    if f["dur"] > 12.0:
                        print(f"  [skip-long] {cid} {ezh} c{ci} dur={f['dur']}", flush=True)
                        continue
                    if best is None or sc > best[0]:
                        best = (sc, seed, w, sr, f)
                except Exception as ex:
                    print(f"  [cand] {cid} {ezh} c{ci} FAIL: {ex}", flush=True)

            if best is None:
                print(f"[FAIL] {cid} {ezh}", flush=True)
                report.append({"id": cid, "emotion": ezh, "emotion_key": ek, "ok": False})
                continue

            sc, seed, w, sr, f = best
            sf.write(path, w, sr)
            print(f"[ok] {cid} {ezh} score={sc:.1f} seed={seed} dur={f['dur']}s "
                  f"f0={f['f0_mean']} rms={f['rms_db']} rate={f['rate']}", flush=True)
            report.append({"id": cid, "name": c.get("name", cid), "emotion": ezh,
                           "emotion_key": ek, "path": path, "ok": True,
                           "score": round(sc, 1), "seed": seed, "base_f0": base_f0,
                           "best_feats": f, "candidates": cands})

    out = f"/ssd2/duanju/voices/japanese_ja_voices_shard{args.shard}.json"
    json.dump({"shard": args.shard, "of": args.of, "count": len(report), "items": report},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    # 基线音高必须落盘：否则下次运行读不到 base_f0，
    # f0 相对基线判定与性别漂移检查全部失效，选优会静默退化。
    bf_path = os.path.join(BASE_F0_DIR, f"base_shard{args.shard}_of{args.of}.json")
    json.dump(base_map, open(bf_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"[base_f0] saved {len(base_map)} -> {bf_path}", flush=True)
    ok = [r for r in report if r.get("ok")]
    avg = sum(r["score"] for r in ok) / max(len(ok), 1)
    print(f"[done] shard{args.shard}: {len(ok)}/{len(report)} avg_score={avg:.1f} -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
