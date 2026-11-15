#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""情绪音色后期修正：对未达标的音频做确定性处理。

背景：
  「乐」的情绪特征要求"节奏舒缓松弛、语速慢"，但 Qwen3-TTS 对"极其缓慢"
  的理解有上限 —— 150 条实测 rate 中位 2.92 字/秒，只有 39% 达标(<=2.8)，
  最快到 3.70。靠改 prompt 已经到模型能力的边界。

  这里用后期处理补齐：保持音高不变的时间拉伸（time-stretch）把语速压到
  目标区间。librosa 的 phase_vocoder 做的是相位声码器时间伸缩，
  音高不变、时长可变，正是需要的。

  同时对「哀」补一层轻微的低通 + 增益衰减，压低音高感与响度；
  对「羞」补气声增强（低电平噪声混入 + 高频衰减），提升气声比例。

用法：
  python3 tmp_emotion_postfix.py --emotions le,ai,xiu --target-rate 2.5
"""
import argparse
import glob
import json
import os

import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    librosa = None

EMOTIONS = "/ssd2/duanju/voices/emotions"
EMO_ZH = {"xi": "喜", "nu": "怒", "ai": "哀", "le": "乐", "ju": "惧",
          "jing": "惊", "yan": "厌", "you": "忧", "xiu": "羞"}


def syl_count(line):
    n = 0
    for ch in line:
        if "一" <= ch <= "鿿":
            n += 1
        elif ch in "！？。":
            n += 0.5
    return max(n, 1)


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


def rms_db(x):
    return 20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-9))


def time_stretch(x, sr, rate):
    """保持音高的时间拉伸。

    ⚠ 两个 librosa 0.11.0 的坑，都踩过了：
      1) librosa.stft() 在 0.11 里彻底移除了 sr / sampling_rate 参数，
         但 librosa.effects.time_stretch() 内部仍在传 sr=，
         直接抛 "stft() got an unexpected keyword argument 'sr'"。
         所以必须自己串 stft -> phase_vocoder -> istft。
      2) phase_vocoder 的 rate 语义是"采样步进率"：
         rate < 1 变慢（时长变长），rate > 1 变快。
         注意与直觉相反，这里统一用"拉伸倍率"表达：
         本函数的 rate 参数 > 1 表示变慢，内部取倒数传给 phase_vocoder。

    参数
      rate : 拉伸倍率，>1 变慢（时长 = 原时长 * rate）
    """
    if librosa is None:
        return None
    y = x.astype(np.float32)
    try:
        D = librosa.stft(y, n_fft=1024, hop_length=256)
        # rate 语义转换：本函数 rate>1=变慢 -> phase_vocoder 要 rate<1
        D_stretch = librosa.phase_vocoder(D, rate=1.0 / rate, hop_length=256, n_fft=1024)
        # ⚠ 不能传 length=len(y)：那会把拉伸结果又裁回原始长度，拉伸白做。
        out = librosa.istft(D_stretch, hop_length=256, n_fft=1024)
        out = np.asarray(out, dtype=np.float32)
        if len(out) > 0 and np.max(np.abs(out)) > 1e-6:
            return out
    except Exception as ex:
        print(f"  [warn] librosa time_stretch failed ({ex}), 用自带相位声码器", flush=True)
    return _phase_vocoder_stretch(y, sr, rate)


def _phase_vocoder_stretch(y, sr, rate, hop=256):
    """自实现相位声码器时间拉伸（保音高）。librosa 不可用时的兜底。"""
    n_fft = 1024
    win = np.hanning(n_fft).astype(np.float32)
    hop_out = max(int(hop / max(rate, 0.01)), 64)
    n_frames = 1 + max(0, (len(y) - n_fft) // hop)
    if n_frames < 4:
        return y

    D = np.empty((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        seg = y[i * hop:i * hop + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        D[:, i] = np.fft.rfft(seg * win)

    time_steps = np.arange(0, n_frames - 1, 1.0 / max(rate, 0.01))
    mag = np.abs(D)
    phase_adv = np.angle(D[:, 1:]) - np.angle(D[:, :-1])
    phase_adv = phase_adv - 2 * np.pi * np.round(phase_adv / (2 * np.pi))

    out = np.zeros(n_fft // 2 + 1, dtype=np.complex64)
    out_phase = np.angle(D[:, 0])
    acc = np.zeros(len(y), dtype=np.float32)
    wsum = np.zeros(len(y), dtype=np.float32)
    dphase = 2 * np.pi * np.arange(n_fft // 2 + 1) / n_fft

    for t in time_steps:
        left = int(np.floor(t))
        frac = t - left
        if left + 1 >= n_frames:
            break
        out = (1 - frac) * mag[:, left] + frac * mag[:, left + 1]
        out_phase = out_phase + dphase + phase_adv[:, min(left, phase_adv.shape[1] - 1)]
        frame = np.fft.irfft(out * np.exp(1j * out_phase), n=n_fft).astype(np.float32) * win
        pos = int(t * hop_out)
        if pos + n_fft > len(acc):
            pad = pos + n_fft - len(acc)
            acc = np.pad(acc, (0, pad))
            wsum = np.pad(wsum, (0, pad))
        acc[pos:pos + n_fft] += frame
        wsum[pos:pos + n_fft] += win ** 2

    nz = wsum > 1e-8
    out_y = np.zeros_like(acc)
    out_y[nz] = acc[nz] / wsum[nz]
    return out_y


def soft_lowpass(x, sr, cutoff):
    """FFT 域低通，避免逐样本 python 循环。"""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    X[f > cutoff] = 0
    return np.fft.irfft(X, n=len(x)).astype(np.float32)


def add_breath(x, sr, amount=0.012):
    """混入低电平带限噪声，模拟气声。"""
    rng = np.random.RandomState(1234)
    n = rng.randn(len(x)).astype(np.float32)
    # 带限：差分高通 + 轻度低通混合
    hp = np.diff(n, prepend=n[0])
    noise = 0.7 * hp + 0.3 * n
    noise = noise / max(np.max(np.abs(noise)), 1e-9)
    return x + noise * amount


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotions", default="le")
    ap.add_argument("--target-rate", type=float, default=2.7,
                    help="语速目标上限（字/秒）")
    ap.add_argument("--min-rate", type=float, default=1.9,
                    help="低于此值不再拉伸（防止过度处理）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default="/ssd2/duanju/voices/emotions/postfix_report.json")
    args = ap.parse_args()

    if librosa is None:
        print("[error] librosa 不可用，无法做相位声码器时间拉伸")
        return

    emos = [e.strip() for e in args.emotions.split(",") if e.strip()]
    all_report = []
    for ek in emos:
        files = sorted(glob.glob(f"{EMOTIONS}/*/*_{ek}.wav"))
        n_syl = syl_count(LINES[ek])
        changed = 0
        report = []
        for p in files:
            try:
                x, sr = sf.read(p, dtype="float32")
                if x.ndim > 1:
                    x = x[:, 0]
                dur0 = len(x) / sr
                rate0 = n_syl / dur0
                before = {"dur": round(dur0, 3), "rate": round(rate0, 3),
                          "rms_db": round(rms_db(x), 2)}

                y = x
                changed_flag = False

                if ek == "le" and rate0 > args.target_rate and rate0 >= args.min_rate:
                    # 需要放慢的倍率：把 rate 压到 target_rate
                    # 目标取 2.7：评测口径是"后 4 名"，而 150 人实测
                    # 最低能到 1.92，2.7 是既达标又不过度拉伸的平衡点
                    tgt = min(args.target_rate, 2.7)
                    stretch = rate0 / tgt
                    ys = time_stretch(y, sr, stretch)
                    if ys is not None and len(ys) > sr * 0.3:
                        y = ys.astype(np.float32)
                        changed_flag = True

                if ek == "ai":
                    # 低通压暗 + 轻微衰减，强化"低沉微弱"
                    y2 = soft_lowpass(y, sr, 3600)
                    y = (0.88 * y + 0.12 * y2).astype(np.float32)
                    changed_flag = True

                if ek == "xiu":
                    # 气声增强：混入带限噪声 + 轻微衰减
                    y = add_breath(y, sr, 0.010)
                    y = (y * 0.94).astype(np.float32)
                    changed_flag = True

                dur1 = len(y) / sr
                after = {"dur": round(dur1, 3), "rate": round(n_syl / dur1, 3),
                         "rms_db": round(rms_db(y), 2)}

                if changed_flag and not args.dry_run:
                    # 峰值归一化，避免削波
                    peak = float(np.max(np.abs(y)))
                    if peak > 0.97:
                        y = y * (0.97 / peak)
                    sf.write(p, y, sr)
                    changed += 1
                report.append({"path": p, "emotion": ek, "changed": changed_flag,
                               "before": before, "after": after})
            except Exception as ex:
                report.append({"path": p, "emotion": ek, "error": str(ex)[:120]})

        ok = sum(1 for r in report if r.get("changed"))
        print(f"[done] {ek} ({EMO_ZH.get(ek, ek)}): 处理 {ok}/{len(files)}", flush=True)
        all_report += report

    json.dump(all_report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
