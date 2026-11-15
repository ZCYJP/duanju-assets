#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为所有角色生成 9 种情绪音色（喜/怒/哀/乐/惧/惊/厌/忧/羞）。

情绪声学特征表来自需求定义：
  喜  音调偏高、节奏轻快跳跃、富有弹性；清脆笑声
  怒  音量极大、重音明显、语速急促有爆发力；怒吼咆哮
  哀  音调低沉、音量微弱、语速缓慢带颤抖；啜泣抽泣
  乐  音调柔和平稳、音量适中、节奏舒缓松弛；满足轻哼
  惧  音调发尖或发哑、呼吸紊乱、明显颤抖；尖叫结巴
  惊  极具突发性、短促吸气、音调瞬间大幅上扬；惊呼
  厌  发声偏鼻腔或喉咙后部、带摩擦感、语速慢；撇嘴冷哼
  忧  音量偏轻、节奏不均（停顿多）、气息绵长；欲言又止
  羞  音量极小、气声重、小心翼翼、犹豫停顿；支支吾吾

每个情绪用一段贴合情绪的台词 + 该情绪的音色描述，
在角色原本 voice_desc 的基础上叠加情绪特征，保持音色辨识度。

输出：/ssd2/duanju/voices/emotions/<CID>/<CID>_<emo>.wav
"""
import argparse, json, os, sys
import numpy as np
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

OUT = "/ssd2/duanju/voices/emotions"
DESIGN = os.environ.get("QWEN_TTS_DESIGN", "/ssd2/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
CAST = os.environ.get("CAST_JSON", "/ssd2/duanju/scripts/cast_all_150.json")

# key, 序号, 中文名, 台词, 情绪声学描述
EMOTIONS = [
    ("xi", 1, "喜",
     "太好了！哈哈，这下可让我高兴坏了，真是让人忍不住笑出声来！",
     "开心的声音，音调偏高，节奏轻快跳跃，富有弹性，音量饱满，"
     "笑声明脆爽朗带有明显的欢呼感，整体语调上扬，语速轻快，"
     "结尾带着兴奋的高音"),
    ("nu", 2, "怒",
     "够了！你别再说了，这件事我绝不会让步——滚出去！",
     "愤怒的声音，音量极大，重音明显，语速急促有爆发力，"
     "音调强硬而不稳定，带咆哮和咬牙的感觉，胸腔共鸣重，"
     "气息粗重，句尾像是从喉咙里吼出来的"),
    ("ai", 3, "哀",
     "怎么会这样……我以为一切都还来得及。唉，现在说什么都晚了。",
     "悲伤的声音，音调低沉，音量微弱，语速缓慢并带颤抖，"
     "气息断续而不稳，带明显的哽咽和啜泣感，"
     "喉音发紧，尾音无力地下垂，偶有长长的叹息"),
    ("le", 4, "乐",
     "嗯，这样挺好的。安安静静地待一会儿，心里很舒坦。",
     "平静安逸的声音，音调柔和平稳，音量适中，节奏舒缓松弛，"
     "气息平和悠长，带满足的轻哼和轻松的呼吸，"
     "语调几乎没有起伏，温暖而放松"),
    ("ju", 5, "惧",
     "别、别过来……我、我什么都没看见……你别再靠近了……",
     "害怕紧张的声音，音调发尖且发哑，呼吸紊乱，"
     "明显颤抖并伴随结巴，音量忽高忽低，"
     "带有倒吸凉气和牙齿打颤的细微声响，气息短促破碎"),
    ("jing", 6, "惊",
     "啊？！这、这怎么可能？哇！怎么会是这样……",
     "惊讶的声音，极具突发性，起音前有短促的吸气，"
     "音调瞬间大幅上扬，音量骤然放大，"
     "带有惊呼和倒吸冷气的感觉，短促而尖锐，"
     "随后带着难以置信的颤抖"),
    ("yan", 7, "厌",
     "啧……真是的，算了，我不想再说了。噫，别碰那个东西。",
     "厌恶嫌弃的声音，发声偏鼻腔或喉咙后部，带明显摩擦感，"
     "语速缓慢而拖沓，音调低沉偏平，带着撇嘴的啧声"
     "和鼻腔冷哼的鼻音，尾音拖长并带有嫌弃的意味"),
    ("you", 8, "忧",
     "唉……我一直在想，这件事到底该怎么办……可是……到底该怎么办呢……",
     "焦虑纠结的声音，音量偏轻，节奏不均停顿很多，"
     "气息绵长，语调起伏犹豫，"
     "带有欲言又止和喃喃自语的特征，"
     "字与字之间拉长并伴随长长的叹气"),
    ("xiu", 9, "羞",
     "那个……其实我……嗯……没什么啦……你、你别看我……",
     "害羞委屈的声音，音量极小，气声很重，"
     "小心翼翼并带有犹豫停顿，语速偏慢，"
     "带有细小的回答、尴尬干笑和吸鼻子的气音，"
     "音调略微上扬但马上又收回去，显得局促不安"),
]


def rough_f0(x, sr):
    frame = int(sr * 0.04)
    hop = int(sr * 0.02)
    vals = []
    for st in range(0, len(x) - frame, hop):
        seg = x[st:st + frame].astype(np.float64)
        if np.sqrt(np.mean(seg ** 2)) < 0.012:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, mode="full")[frame - 1:]
        lo, hi = int(sr / 400), int(sr / 70)
        if hi >= len(ac):
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        vals.append(sr / lag)
    return float(np.median(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--of', type=int, default=1)
    ap.add_argument('--only', default='', help='只生成指定角色ID（逗号分隔）')
    args = ap.parse_args()

    cast = json.load(open(CAST, encoding="utf-8"))
    if args.only:
        want = {x.strip() for x in args.only.split(',') if x.strip()}
        cast = [c for c in cast if c['id'] in want]
    mine = [(i, c) for i, c in enumerate(cast) if i % args.of == args.shard]
    print(f"[load] shard {args.shard}/{args.of}: {len(mine)} roles x {len(EMOTIONS)} emotions", flush=True)

    model = Qwen3TTSModel.from_pretrained(
        DESIGN, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")

    report = []
    for gi, c in mine:
        cid, name = c['id'], c['name']
        base_desc = c.get('voice_desc', '')
        # 在角色原本音色描述上叠加情绪特征：保留音色辨识度，只改情绪表现
        base_core = base_desc.split('，')[0] if base_desc else ''
        cdir = os.path.join(OUT, cid)
        os.makedirs(cdir, exist_ok=True)
        for ei, (ek, eno, ezh, line, edesc) in enumerate(EMOTIONS):
            path = os.path.join(cdir, f'{cid}_{ek}.wav')
            desc = f"{base_core}，用{ezh}的情绪演绎这句话：{edesc}"
            seed = 20000 + gi * 97 + ei * 13
            if os.path.exists(path) and os.path.getsize(path) > 10000:
                try:
                    w0, sr0 = sf.read(path, dtype="float32")
                    if w0.ndim > 1:
                        w0 = w0[:, 0]
                    f0 = rough_f0(w0, sr0)
                    dur = len(w0) / sr0
                    report.append({"id": cid, "name": name, "emotion": ezh, "emotion_key": ek,
                                   "emotion_no": eno, "path": path, "dur": round(dur, 2),
                                   "f0": round(f0, 1), "ok": True, "cached": True})
                    continue
                except Exception:
                    pass
            try:
                torch.manual_seed(seed)
                wavs, sr = model.generate_voice_design(line, desc, "Chinese")
                w = np.asarray(wavs[0], dtype=np.float32)
                sf.write(path, w, sr)
                f0 = rough_f0(w, sr)
                dur = len(w) / sr
                peak = float(np.max(np.abs(w)))
                report.append({"id": cid, "name": name, "emotion": ezh, "emotion_key": ek,
                               "emotion_no": eno, "path": path, "dur": round(dur, 2),
                               "f0": round(f0, 1), "peak": round(peak, 3),
                               "seed": seed, "ok": True})
                print(f"[ok] {cid} {name} {ezh} {dur:.1f}s F0≈{f0:.0f}Hz", flush=True)
            except Exception as ex:
                print(f"[FAIL] {cid} {ezh}: {ex}", flush=True)
                report.append({"id": cid, "name": name, "emotion": ezh, "emotion_key": ek,
                               "emotion_no": eno, "path": path, "ok": False, "error": str(ex)})

    mp = f"/ssd2/duanju/voices/emotion_manifest_shard{args.shard}.json"
    json.dump({"host": "osa00-h100-038", "model": DESIGN, "shard": args.shard,
               "of": args.of, "count": len(report), "items": report},
              open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ok = sum(1 for r in report if r.get("ok"))
    print(f"[done] shard{args.shard}: {ok}/{len(report)} emotion voices -> {mp}", flush=True)


if __name__ == "__main__":
    main()
