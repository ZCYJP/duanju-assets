#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""9 情绪音色声学特征量化核验。

对照用户给定的 9 种情绪声音特征表，逐条检查生成音频是否落在预期区间：
  1 喜 xi    音调偏高、节奏轻快跳跃、富有弹性
  2 怒 nu    音量极大、重音明显、语速急促有爆发力
  3 哀 ai    音调低沉、音量微弱、语速缓慢带颤抖
  4 乐 le    音调柔和平稳、音量适中、节奏舒缓松弛
  5 惧 ju    音调发尖或发哑、呼吸紊乱、明显颤抖
  6 惊 jing  极具突发性、短促吸气、音调瞬间大幅上扬
  7 厌 yan   发声偏鼻腔或喉咙后部、带摩擦感、语速慢
  8 忧 you   音量偏轻、节奏不均（停顿多）、气息绵长
  9 羞 xiu   音量极小、气声重、小心翼翼、犹豫停顿

量化指标（每条音频）：
  f0_mean / f0_std   基频均值与抖动（颤音/颤抖的客观量）
  rms_db             整体响度
  p95_db             峰值响度（爆发力）
  dur                时长（语速的粗略代理：同文本下时长越短语速越快）
  zcr                过零率（高频能量占比，尖锐/嘶哑的代理）
  crest              峰均比（动态起伏，爆发/突增的代理）
  onset_jerk         起始抖动（惊的突发性）
  breath_ratio       气声比（低能量段占比，羞的气声）

判定：先用全体均值做同情绪内的相对排序（该情绪的音高是否确实高于/低于
其他情绪），这比绝对阈值稳健得多；同时输出每情绪的均值表供人工核对。

用法：python3 tmp_emotion_audio_qc.py [--root ...] [--out ...]
"""
import argparse
import glob
import json
import os
import re
import wave

import numpy as np

EMO = ["xi", "nu", "ai", "le", "ju", "jing", "yan", "you", "xiu"]
EMO_ZH = {"xi": "喜", "nu": "怒", "ai": "哀", "le": "乐", "ju": "惧",
          "jing": "惊", "yan": "厌", "you": "忧", "xiu": "羞"}

# 各情绪的台词（与 gen_emotion_voices.py 一致），用于把绝对时长归一化成语速。
# 不同情绪台词长度差异很大（21~33 字），直接比 dur 会被文本长度污染，
# 必须换算成"字/秒"才是可横向比较的语速指标。
LINES = {
    "xi":   "太好了！哈哈，这下可让我高兴坏了，真是让人忍不住笑出声来！",
    "nu":   "够了！你别再说了，这件事我绝不会让步——滚出去！",
    "ai":   "怎么会这样……我以为一切都还来得及。唉，现在说什么都晚了。",
    "le":   "嗯，这样挺好的。安安静静地待一会儿，心里很舒坦。",
    "ju":   "别、别过来……我、我什么都没看见……你别再靠近了……",
    "jing": "啊？！这、这怎么可能？哇！怎么会是这样……",
    "yan":  "啧……真是的，算了，我不想再说了。噫，别碰那个东西。",
    "you":  "唉……我一直在想，这件事到底该怎么办……可是……到底该怎么办呢……",
    "xiu":  "那个……其实我……嗯……没什么啦……你、你别看我……",
}


def syl_count(line):
    """近似音节数：汉字算 1，非汉字（标点/省略号）折半并至少计 1。"""
    n = 0
    for ch in line:
        if "一" <= ch <= "鿿":
            n += 1
        elif ch in "！？。":
            n += 0.5
    return max(n, 1)


def read_wav(p):
    with wave.open(p, "rb") as w:
        nch, sw, sr, nf = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(nf)
    if sw == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
        x /= 128.0
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    return x, sr


def f0_track(x, sr):
    """自相关法粗提取基频（每 40ms 一帧）。"""
    hop = int(sr * 0.04)
    win = int(sr * 0.04)
    lo, hi = int(sr / 400), int(sr / 70)          # 70Hz ~ 400Hz
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


def features(p):
    x, sr = read_wav(p)
    if len(x) < sr * 0.1:
        return None
    dur = len(x) / sr
    rms = float(np.sqrt(np.mean(x ** 2)))
    rms_db = 20 * np.log10(max(rms, 1e-9))
    peak = float(np.max(np.abs(x)))
    p95 = float(np.percentile(np.abs(x), 95))
    p95_db = 20 * np.log10(max(p95, 1e-9))
    crest = float(peak / max(rms, 1e-9))

    # 过零率（高频能量代理）
    zcr = float(np.mean(np.abs(np.diff(np.sign(x))) > 0))

    f0 = f0_track(x, sr)
    f0_mean = float(np.median(f0)) if len(f0) else 0.0
    f0_std = float(np.std(f0)) if len(f0) else 0.0

    # 起始抖动：前 150ms 的能量包络斜率与变化率
    head = x[:int(sr * 0.15)]
    if len(head) > 16:
        env = np.abs(head)
        onset_jerk = float(np.max(np.diff(env)) / max(np.mean(env), 1e-9))
    else:
        onset_jerk = 0.0

    # 气声比：低于 20% 峰值的帧占比
    hop = int(sr * 0.02)
    frames = x[:len(x) // hop * hop].reshape(-1, hop)
    if len(frames):
        fe = np.sqrt(np.mean(frames ** 2, axis=1))
        thr = 0.2 * (np.percentile(fe, 95) + 1e-9)
        breath_ratio = float(np.mean(fe < thr))
    else:
        breath_ratio = 0.0

    return {
        "dur": round(dur, 3),
        "rms_db": round(rms_db, 2),
        "p95_db": round(p95_db, 2),
        "crest": round(crest, 3),
        "zcr": round(zcr, 4),
        "f0_mean": round(f0_mean, 1),
        "f0_std": round(f0_std, 1),
        "onset_jerk": round(onset_jerk, 3),
        "breath_ratio": round(breath_ratio, 3),
    }


def add_rate(feat, emo):
    """把绝对时长换算成归一化语速（音节/秒）。"""
    n = syl_count(LINES[emo])
    feat["rate"] = round(n / max(feat["dur"], 1e-6), 3)
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/ssd2/duanju/voices/emotions")
    ap.add_argument("--out", default="/ssd2/duanju/voices/emotions/audio_qc.json")
    ap.add_argument("--sample", type=int, default=30, help="每情绪抽样数")
    ap.add_argument("--cast", default="/ssd2/duanju/scripts/cast_all_150.json")
    args = ap.parse_args()

    # 角色性别表：用于分层评测
    cast = json.load(open(args.cast, encoding="utf-8"))
    gender_of = {c["id"]: c.get("gender", "") for c in cast}

    rng = np.random.RandomState(20261115)
    data = {}
    for e in EMO:
        files = sorted(glob.glob(f"{args.root}/*/*_{e}.wav"))
        if not files:
            print(f"[warn] no files for {e}")
            continue
        if len(files) > args.sample:
            idx = rng.choice(len(files), args.sample, replace=False)
            files = [files[i] for i in sorted(idx)]
        feats = []
        for p in files:
            try:
                f = features(p)
            except Exception as ex:
                f = None
            if f:
                cid = os.path.basename(p).split("_")[0]
                f["cid"] = cid
                f["gender"] = gender_of.get(cid, "")
                feats.append(add_rate(f, e))
        data[e] = feats
        print(f"[done] {e} ({EMO_ZH[e]}): {len(feats)} samples", flush=True)

    summary = {}
    for e, feats in data.items():
        if not feats:
            continue
        agg = {}
        for k in feats[0]:
            if k in ("cid", "gender"):
                continue
            v = np.array([f[k] for f in feats], dtype=float)
            agg[k] = {"mean": round(float(v.mean()), 4), "std": round(float(v.std()), 4),
                      "p10": round(float(np.percentile(v, 10)), 4),
                      "p90": round(float(np.percentile(v, 90)), 4)}
        # 按性别分层统计
        by_g = {}
        for g in ("男", "女"):
            sub = [f for f in feats if f.get("gender") == g]
            if not sub:
                continue
            gm = {}
            for k in ("f0_mean", "rms_db", "rate", "f0_std", "zcr", "crest",
                      "onset_jerk", "breath_ratio"):
                v = np.array([f[k] for f in sub], dtype=float)
                gm[k] = round(float(v.mean()), 4)
            by_g[g] = {"n": len(sub), "mean": gm}
        summary[e] = {"zh": EMO_ZH[e], "n": len(feats), "stats": agg, "by_gender": by_g}

    json.dump({"per_file": data, "summary": summary}, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # -------- 相对排序判定 --------
    print("\n=== 情绪均值表（每项前 3 名 = 该指标最突出的情绪） ===")
    metrics = ["f0_mean", "rms_db", "p95_db", "rate", "zcr", "crest", "f0_std",
               "onset_jerk", "breath_ratio"]
    labels = {"f0_mean": "音高", "rms_db": "响度", "p95_db": "峰值", "rate": "语速(字/秒)",
              "zcr": "高频/尖锐", "crest": "动态起伏", "f0_std": "颤音抖动",
              "onset_jerk": "起始突发", "breath_ratio": "气声比"}
    rank = {}
    for m in metrics:
        vals = {e: summary[e]["stats"][m]["mean"] for e in summary if m in summary[e]["stats"]}
        order = sorted(vals, key=lambda k: vals[k], reverse=True)
        rank[m] = order
        top3 = ", ".join(f"{EMO_ZH[e]}({vals[e]:.2f})" for e in order[:3])
        bot3 = ", ".join(f"{EMO_ZH[e]}({vals[e]:.2f})" for e in order[-3:])
        print(f"{labels[m]:>12}: 高= {top3}")
        print(f"{'':>12}    低= {bot3}")

    # 期望（严格对照用户给定的 9 种情绪声音特征表）
    # 每项：指标 -> ("high"|"low"|"high_or_low", 期望名次区间)
    #   high  = 该情绪应在此指标上排进前 N
    #   low   = 该情绪应在此指标上排进后 N
    #   high_or_low = 极端值（偏高或偏低都算符合）
    expect = {
        "xi":   {"f0_mean": ("high", 3), "rate": ("high", 4), "rms_db": ("high", 4)},
        "nu":   {"rms_db": ("high", 4), "p95_db": ("high", 4), "rate": ("high", 4),
                 "crest": ("high", 4)},
        "ai":   {"f0_mean": ("low", 4), "rms_db": ("low", 4), "rate": ("low", 4),
                 "f0_std": ("high", 4)},
        "le":   {"rate": ("low", 4), "rms_db": ("mid", 6)},
        "ju":   {"f0_mean": ("high_or_low", 0), "zcr": ("high", 4),
                 "f0_std": ("high", 4), "crest": ("high", 4)},
        "jing": {"onset_jerk": ("high", 4), "f0_mean": ("high", 4), "rate": ("high", 4)},
        "yan":  {"rate": ("low", 5), "rms_db": ("low", 4), "zcr": ("high", 4)},
        "you":  {"rms_db": ("low", 4), "rate": ("low", 4), "crest": ("high", 4)},
        "xiu":  {"rms_db": ("low", 4), "breath_ratio": ("high", 4), "rate": ("low", 5)},
    }
    print("\n=== 期望 vs 实测（对照用户声音特征表） ===")
    print("【分层评测】男/女各自在同性别内部排名 ——")
    print("  音高、响度等绝对声学量受性别影响极大（男声基线 ~117Hz、")
    print("  女声基线 ~214Hz），混在一起排名会把「哀」这类低沉情绪误判。\n")
    verdicts = {}
    total_ok = total_all = 0
    for e, exp in expect.items():
        if e not in summary:
            continue
        notes = []
        ok_n = 0
        for m, (want, span) in exp.items():
            # 分层：在同性别内部取该情绪的排名
            pos_by_g = {}
            for g in ("男", "女"):
                vals = {x: summary[x]["by_gender"][g]["mean"][m]
                        for x in summary
                        if g in summary[x].get("by_gender", {}) and
                        m in summary[x]["by_gender"][g]["mean"]}
                if not vals:
                    continue
                order = sorted(vals, key=lambda k: vals[k], reverse=True)
                pos_by_g[g] = (order.index(e), len(order), vals[e])
            if not pos_by_g:
                continue
            total_all += 1
            good = True
            detail = []
            for g, (pos, n, v) in pos_by_g.items():
                if want == "high":
                    g_good = pos < span
                    tag = f"前{span}"
                elif want == "low":
                    g_good = pos >= n - span
                    tag = f"后{span}"
                elif want == "high_or_low":
                    g_good = pos < 3 or pos >= n - 3
                    tag = "极端"
                else:  # mid
                    g_good = span * 2 > pos >= span / 2
                    tag = "中段"
                detail.append(f"{g}#{pos+1}/{n}({tag},v={v:.2f})")
                if not g_good:
                    good = False
            if good:
                ok_n += 1
                total_ok += 1
                notes.append(f"{labels[m]}✓")
            else:
                notes.append(f"{labels[m]}✗[" + " ".join(detail) + "]")
        verdicts[e] = {"pass": f"{ok_n}/{len(exp)}", "notes": notes}
        print(f"{EMO_ZH[e]}({e}) {ok_n}/{len(exp)}: " + " | ".join(notes))

    print(f"\n[总评·分层] {total_ok}/{total_all} 项符合用户声音特征表 "
          f"({total_ok/total_all*100:.0f}%)", flush=True)

    json.dump({"summary": summary, "rank": rank, "verdicts": verdicts},
              open(args.out.replace(".json", "_verdict.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
