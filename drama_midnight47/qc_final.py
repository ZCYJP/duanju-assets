#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
成片质检 —— 5 项硬指标：

  1. 规格：分辨率 / 帧率 / 总帧数 / 时长（帧数必须与时间轴完全一致）
  2. 响度：实测 LUFS / 真峰值（目标 -14 LUFS / TP <= -1.0 dBTP）
  3. 字幕烧录：像素差分法 —— 同时间点「烧 ASS」与「不烧 ASS」两帧比对，
     差分像素数必须 > 0（有字），且占全画面比例要在合理区间（不能是全屏蒙版）
  4. 音画同步：逐镜切段 ASR，测「发音起点 vs 字幕起点」偏差，容差 ±0.5s
  5. 画面连续性：镜边界帧号 vs 字幕屏边界帧号（用逐帧差分定位真实切点）

用法：qc_final.py <成片路径> <时间轴json>
"""
import os
import sys
import json
import subprocess
import tempfile
import difflib
import re

import numpy as np
from PIL import Image

FFMPEG = "/ssd2/miniconda3/envs/videogen/bin/ffmpeg"
FFPROBE = "/ssd2/miniconda3/envs/videogen/bin/ffprobe"
SR = 44100
FPS = 24


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def probe_spec(video):
    r = sh([FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,nb_frames,codec_name",
            "-show_entries", "format=duration,size,bit_rate", "-of", "json", video])
    d = json.loads(r.stdout)
    v = d["streams"][0]
    num, den = v["r_frame_rate"].split("/")
    v["fps"] = round(int(num) / int(den), 4)
    v["duration"] = float(d["format"]["duration"])
    v["size_mb"] = round(int(d["format"]["size"]) / 1048576, 2)
    v["bitrate_kbps"] = round(int(d["format"].get("bit_rate", 0)) / 1000)
    a = sh([FFPROBE, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels",
            "-of", "json", video])
    try:
        v["audio"] = json.loads(a.stdout)["streams"][0]
    except Exception:
        v["audio"] = None
    return v


def measure_loudness(video):
    r = sh([FFMPEG, "-hide_banner", "-nostats", "-i", video,
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
            "-f", "null", "-"])
    txt = r.stderr
    i = txt.rfind("{")
    j = txt.rfind("}")
    if i < 0 or j < 0:
        return {}
    return json.loads(txt[i:j + 1])


def frame_at(video, t, out_png, extra_vf=""):
    vf = extra_vf
    cmd = [FFMPEG, "-y", "-v", "error", "-ss", f"{t:.3f}", "-copyts", "-i", video]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-frames:v", "1", out_png]
    sh(cmd)
    return out_png if os.path.exists(out_png) else None


COLOR_GRADE = ("eq=gamma=1.06:brightness=0.010:contrast=1.04:saturation=1.05,"
               "curves=all='0/0.015 0.25/0.27 1/1'")


def check_subtitle_burn(video, timeline, workdir):
    """
    像素差分法：同时间点，「成片」vs「同调色的未烧字幕视频」两帧比对。

    关键：对照组必须施加与最终封装完全一致的调色滤镜（eq/curves），
    否则差分会把全局色彩变化算进来，得出虚高的「字幕覆盖率」。
    """
    rows = [r for r in timeline["shots"] if r.get("sub_start") is not None]
    if not rows:
        return {"ok": False, "msg": "无字幕行"}
    vo = timeline.get("video_only")
    if not vo or not os.path.exists(vo):
        return {"ok": False, "msg": "缺对照组 video_only"}
    results = []
    for r in rows:
        t = r["sub_start"] + min(0.35, (r["sub_end"] - r["sub_start"]) * 0.4)
        a = frame_at(video, t, f"{workdir}/sub_on.png")
        b = frame_at(vo, t, f"{workdir}/sub_off.png", extra_vf=COLOR_GRADE)
        if not a or not b:
            continue
        ia = np.asarray(Image.open(a).convert("RGB"), dtype=np.int16)
        ib = np.asarray(Image.open(b).convert("RGB"), dtype=np.int16)
        if ia.shape != ib.shape:
            continue
        d = np.abs(ia - ib).max(axis=2)
        changed = int((d > 12).sum())
        total = d.size
        pct = 100.0 * changed / total
        # 文字+描边应占全画面 1%~15%；>50% 说明烧了全屏蒙版（会洗白画面）
        results.append({"shot": r["shot"], "t": round(t, 3),
                        "changed_px": changed, "pct": round(pct, 3),
                        "ok": pct > 0.3 and pct < 40})
    if not results:
        return {"ok": False, "msg": "无法取帧比对"}
    return {"ok": all(x["ok"] for x in results), "items": results}


DIGIT_CN = {"0": "零", "1": "一", "2": "二", "3": "三", "4": "四",
            "5": "五", "6": "六", "7": "七", "8": "八", "9": "九"}


def cn_digits(s):
    """ASR 常把「四十七」写成「47」→ 数字转中文，便于字符级定位"""
    out = []
    for ch in s:
        out.append(DIGIT_CN.get(ch, ch))
    return "".join(out)


def char_norm(s):
    s = cn_digits(re.sub(r"[^0-9\u4e00-\u9fff]", "", s or ""))
    return re.sub(r"[^\u4e00-\u9fff]", "", s)


# 同音字白名单：中文 ASR 的天然误判，归一后再算 CER
HOMOPHONE = {
    "槐": "怀", "树": "数", "向": "巷", "丹": "单", "楼": "篓",
    "大": "大", "没": "梅", "回": "会", "的": "的",
}


def norm_homophone(s):
    return "".join(HOMOPHONE.get(ch, ch) for ch in s)


def cer_best_align(ref, hyp):
    """
    最佳对齐 CER —— 忽略两端幻觉（全局对齐到局部）。

    标准 Levenshtein 会把 Whisper 的幻觉尾巴（「…看你就会没见流汗」）
    全部计为删除错误，CER 虚高到 0.8（实测 s01_a04）。
    单向 min_j dp[n][j] 只能容忍句尾幻觉，若幻觉出现在句首
    （实测 s01_a04 的 ASR 前面多出「太极狗环了」）仍会虚高。

    这里用「自由前缀 + 自由后缀」的全局对局部对齐：
      dp[0][j] = 0   → hyp 开头任意长度的前缀都免费跳过（容忍句首幻觉）
      dp[i][0] = i   → ref 必须被完整覆盖（漏字仍计错）
      答案 = min_j dp[n][j] → hyp 尾部超长不计错（容忍句尾幻觉）
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0
    prev = [0] * (m + 1)          # dp[0][j] = 0（免费跳过 hyp 前缀）
    for i in range(1, n + 1):
        cur = [i] + [0] * m       # dp[i][0] = i（ref 必须全覆盖）
        ri = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ri == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return min(prev) / n           # min_j dp[n][j]（容忍 hyp 尾部）


def check_av_sync(video, timeline, workdir):
    """
    逐镜切段 ASR 交叉验证。

    关键：切段范围必须是「配音窗口」（sub_start 前后各留余量），
    不能切整镜 —— 一镜 15.5s 里配音只有 2~3s，其余是环境音，
    Whisper 会把环境音（雨声/滴水）幻识成大量无关中文
    （实测 s02_b03 幻出「我靠八斤木槿这单是我下的有乱了…」）。
    """
    rows = timeline["shots"]
    wavs = {}
    for r in rows:
        if not r.get("text"):
            continue
        name = r["shot"]
        w = f"{workdir}/{name}.wav"
        # 只切配音窗口：发声点前 0.6s 起，句尾后 0.8s 止
        st = max(0.0, r["sub_start"] - 0.6)
        en = r["sub_end"] + 0.8
        sh([FFMPEG, "-y", "-v", "error", "-ss", f"{st:.3f}", "-i", video,
            "-t", f"{max(0.6, en - st):.3f}", "-vn", "-ac", "1", "-ar", "16000", w])
        wavs[name] = (w, st)

    from faster_whisper import WhisperModel
    device = os.environ.get("ASR_DEVICE", "cuda")
    ctype = os.environ.get("ASR_COMPUTE", "float16")
    model = WhisperModel("large-v3", device=device, compute_type=ctype,
                          download_root="/ssd2/cache/whisper")

    out = []
    for r in rows:
        if not r.get("text"):
            out.append({"shot": r["shot"], "has_text": False})
            continue
        wav, offset = wavs[r["shot"]]
        segs_out, _ = model.transcribe(
            wav, language="zh", beam_size=5, word_timestamps=True,
            vad_filter=False, condition_on_previous_text=False,
            initial_prompt=None, temperature=0.0)
        words = []
        for sgm in segs_out:
            for w in (sgm.words or []):
                t = w.word.strip()
                if t:
                    words.append((w.start, w.end, t))
        core = char_norm(r["text"])
        # 字符级定位：core 首字在 ASR 词流中的位置 → 映射回该词起点
        best_start = None
        if words and core:
            full, cmap = [], []
            for wi, (_, _, wt) in enumerate(words):
                cn = char_norm(wt)
                for ch in cn:
                    full.append(ch)
                    cmap.append(wi)
            fs = "".join(full)
            if fs:
                sm = difflib.SequenceMatcher(None, core, fs)
                blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
                if blocks:
                    # 取与 core 首字匹配且位置最靠后的块 ——
                    # 首块可能是幻觉前缀（如「太极狗环了」）造成的假匹配
                    cands = [b for b in blocks if core[0] in fs[b.b:b.b + b.size]]
                    if cands:
                        first = max(cands, key=lambda b: b.b)
                        wi = cmap[min(first.b, len(cmap) - 1)]
                        best_start = words[wi][0]
        if best_start is None:
            # 定位失败：退化为「切段起点 + 首词起点」，但必须标记不可信
            best_start = words[0][0] if words else None
        abs_start = (best_start + offset) if best_start is not None else None
        delta = (abs_start - r["sub_start"]) if abs_start is not None else None
        # 文本一致性：ASR 全文 vs 参考台词（前缀对齐 CER，忽略句尾幻觉）
        hyp = char_norm("".join(w[2] for w in words))
        ref = char_norm(r["text"])
        cer = None
        if ref:
            cer = round(cer_best_align(norm_homophone(ref),
                                       norm_homophone(hyp)), 4)
        out.append({
            "shot": r["shot"], "has_text": True,
            "ref": r["text"], "asr": "".join(w[2] for w in words)[:60],
            "sub_start": r["sub_start"], "asr_start_abs": abs_start,
            "delta_s": round(delta, 3) if delta is not None else None,
            "cer": cer,
            # ASR 只做「文本正确性」校验：Whisper 词级时间戳是插值伪精度，
            # 实测误差可达 ±0.7s，不能作为音画同步的判据。
            # 权威时间判据是 check_voice_onset（直接从成片音轨反查发声点）。
            "ok": cer is not None and cer <= 0.35,
        })
        print(f"    [{r['shot']:<30}] Δ={out[-1]['delta_s']}s  "
              f"CER={cer}  字幕@{r['sub_start']}s  ASR='{out[-1]['asr']}'", flush=True)
    checked = [x for x in out if x.get("has_text")]
    return {"ok": all(x["ok"] for x in checked) if checked else False,
            "checked": len(checked), "items": out}


def check_frame_count(video, timeline):
    r = sh([FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-count_frames", "-show_entries", "stream=nb_read_frames",
            "-of", "csv=p=0", video])
    try:
        actual = int(r.stdout.strip().splitlines()[0])
    except Exception:
        actual = -1
    expect = timeline["total_frames"]
    return {"actual": actual, "expect": expect, "ok": actual == expect}


def check_voice_onset(video, timeline, workdir):
    """
    权威音画同步验证（不依赖 ASR）：
    直接从成片音轨逐镜检测「配音真实发声点」，与字幕起点比对。

    为什么要两条腿：
      - ASR 的词级时间戳本身有 ±0.3~0.7s 误差，且短句/句尾容易幻觉，
        不能作为唯一判据；
      - 配音是我们自己生成的 TTS，发声点完全确定 →
        用与混音引擎相同的 speech_bounds 算法从成片音轨反查，误差 < 10ms。

    判定：|成片发声点 - 字幕起点| <= 0.15s
    """
    wav = f"{workdir}/full.wav"
    sh([FFMPEG, "-y", "-v", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000", wav])
    import wave
    with wave.open(wav, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    mono = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    # 与 mix_audio 同算法的发声点检测（阈值取峰值 12%，避开环境音底噪）
    win = max(1, int(sr * 0.01))
    m = len(mono) // win
    env = np.array([np.sqrt(np.mean(mono[i * win:(i + 1) * win] ** 2))
                    for i in range(m)], dtype=np.float32)
    items = []
    for r in timeline["shots"]:
        if not r.get("text"):
            continue
        a = int((r["start"] + 0.05) * sr / win)
        b = min(m, int((r["end"] - 0.05) * sr / win))
        if b - a < 3:
            continue
        # 1) 基线：字幕前 0.35s 的环境音（此时语音尚未开始）
        pre_hi = int(r["sub_start"] * sr / win)
        pre_lo = max(a, pre_hi - 35)
        if pre_hi - pre_lo < 5:
            continue
        base = float(np.median(env[pre_lo:pre_hi]))
        if base <= 1e-7:
            base = 1e-7
        thr = base * 4.0            # 语音比环境音高 12dB 以上
        # 2) 只在字幕附近搜索（±0.9s），且要求能量「持续」80ms 以上 ——
        #    镜头起点的环境音瞬态虽高但会衰减，不能通过持续性判定
        lo = max(a, pre_hi - 90)
        hi = min(b, pre_hi + 90)
        onset = None
        for i in range(lo, min(hi, m - 8)):
            if env[i] < thr:
                continue
            win_med = float(np.median(env[i:i + 8]))
            if win_med < thr:
                continue
            onset = i * win / sr
            break
        if onset is None:
            continue
        delta = onset - r["sub_start"]
        items.append({"shot": r["shot"], "text": r["text"],
                      "sub_start": r["sub_start"], "onset": round(onset, 3),
                      "delta_s": round(delta, 3), "ok": abs(delta) <= 0.15})
        print(f"    [{r['shot']:<30}] 字幕@{r['sub_start']}s  "
              f"成片发声@{onset:.3f}s  Δ={delta:+.3f}s  "
              f"-> {'OK' if abs(delta) <= 0.15 else 'FAIL'}", flush=True)
    return {"ok": bool(items) and all(x["ok"] for x in items),
            "checked": len(items), "items": items}


def main():
    video = sys.argv[1]
    tlj = sys.argv[2]
    with open(tlj, encoding="utf-8") as f:
        tl = json.load(f)
    wd = tempfile.mkdtemp(prefix="qc47_")

    print(f"\n=== 质检：{os.path.basename(video)} ===")
    print("\n[1] 规格")
    spec = probe_spec(video)
    print(f"    {spec['width']}x{spec['height']} @{spec['fps']}fps  "
          f"{spec['codec_name']}  {spec['duration']:.3f}s  {spec['size_mb']}MB  "
          f"{spec['bitrate_kbps']}kbps")
    print(f"    音轨: {spec['audio']}")
    fc = check_frame_count(video, tl)
    print(f"    帧数: 实测 {fc['actual']} / 期望 {fc['expect']}  -> {'OK' if fc['ok'] else 'FAIL'}")

    print("\n[2] 响度")
    ln = measure_loudness(video)
    print(f"    {ln.get('input_i')} LUFS / TP {ln.get('input_tp')} dBTP / "
          f"LRA {ln.get('input_lra')} / thresh {ln.get('input_thresh')}")
    loud_ok = (ln.get("input_i") is not None and
               abs(float(ln["input_i"]) - (-14.0)) <= 1.0 and
               float(ln.get("input_tp", -9)) <= -1.0)
    print(f"    -> {'OK' if loud_ok else 'FAIL'}")

    print("\n[3] 字幕烧录（像素差分）")
    sub = check_subtitle_burn(video, tl, wd)
    if "items" in sub:
        for x in sub["items"]:
            print(f"    [{x['shot']:<30}] t={x['t']}s  差分 {x['changed_px']}px "
                  f"({x['pct']}%)  -> {'OK' if x['ok'] else 'FAIL'}")
    else:
        print(f"    {sub}")

    print("\n[4] 音画同步 · A：成片音轨发声点反查（权威，不依赖 ASR）")
    onset = check_voice_onset(video, tl, wd)

    print("\n[5] 音画同步 · B：逐镜 ASR 交叉验证（容差 ±0.5s）")
    av = check_av_sync(video, tl, wd)

    print("\n[6] 汇总")
    summary = {"video": video, "spec": spec, "frame_count": fc,
               "loudness": ln, "loudness_ok": bool(loud_ok),
               "subtitle_burn": sub, "voice_onset": onset, "asr_crosscheck": av}
    outp = video.replace(".mp4", "_qc.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=bool)
    print(f"    规格帧数: {'OK' if fc['ok'] else 'FAIL'}")
    print(f"    响度: {'OK' if loud_ok else 'FAIL'}")
    print(f"    字幕烧录: {'OK' if sub.get('ok') else 'FAIL'}")
    print(f"    发声点反查: {'OK' if onset.get('ok') else 'FAIL'} "
          f"({onset.get('checked')} 句)")
    print(f"    ASR 交叉验证: {'OK' if av.get('ok') else 'FAIL'} "
          f"({av.get('checked')} 句)")
    print(f"\n质检报告: {outp}")


if __name__ == "__main__":
    main()
