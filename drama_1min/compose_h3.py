#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《决裂》H3 版终合成 —— 24fps / 原生环境音 + TTS 配音混合。

相比 v1（Wan 16fps）改进：
  1. 24fps 源帧率（H3 原生），画面更顺
  2. 混合 H3 原生环境音（雨声/脚步/空调）与 TTS 配音 —— 这是「短剧感」的关键，
     纯配音 + 静画面 = 用户说的「图片配音」
  3. 人声优先混合：配音期间环境音 duck 到 22%，句间抬回 100%
  4. 精确 60.000s

用法：compose_h3.py [--motion-dir motion_h3_fln] [--out 输出路径]
"""
import os, sys, json, wave, subprocess, shutil, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shots_1min import SHOTS

BASE = "/ssd2/duanju/drama_1min"
VOICE_DIR = f"{BASE}/voice"
OUT_DIR = f"{BASE}/final"
WORK_DIR = f"{BASE}/compose_h3_work"
FONT = "/ssd2/short_drama/fonts/SourceHanSansSC-Bold.otf"

FFMPEG = "/ssd2/miniconda3/envs/videogen/bin/ffmpeg"
FFPROBE = "/ssd2/miniconda3/envs/videogen/bin/ffprobe"

FPS = 24
W, H = 480, 832
SR = 44100
# 镜长分配（总长必须精确 60.000s）：
#   片头 1.0s + 常规镜 11×118帧(4.9167s) + 末镜 124帧(5.1667s) = 60.25s
#   -> 最终输出 -t 60.000 裁掉片尾淡出区 0.25s（无损，末镜台词完整）
SHOT_FRAMES = 118           # 常规镜 118 帧 @24fps = 4.9167s
SHOT_DUR = SHOT_FRAMES / FPS
INTRO_DUR = 1.0             # 片头卡 1.0s
LAST_SHOT_FRAMES = 124      # 末镜完整 124 帧，保证台词不截断
LAST_SHOT_DUR = LAST_SHOT_FRAMES / FPS
TARGET_DUR = 60.0           # 成片精确时长

PAD_HEAD = 0.30             # 镜内配音前置留白
PAD_TAIL = 0.45             # 句间静音垫（铁律 >= 0.45s）
OUTRO_FADE = 1.0

AMBIENT_TARGET_LUFS = -26   # 环境音统一归一化目标（H3 原生音轨跨度达 14.6 LUFS）
AMBIENT_GAIN = 0.35         # 归一化后的环境音增益（垫底）
VOICE_TARGET_RMS = -20.0    # 配音目标 RMS(dBFS)，用确定性增益而非 loudnorm
VOICE_PEAK_CEIL = -3.0      # 配音峰值上限
DUCK_GAIN = 0.25            # 配音期间环境音压低比例

COLOR_SPEAKER = {
    "苏清晚": "&H00E8C4FF&",
    "苏蔓":   "&H0080FFB0&",
    "宋佳宁": "&H00C8D2FF&",
    "裴九渊": "&H004060C0&",
    "陆沉":   "&H00D8D0C0&",
}
COLOR_TEXT = "&H00FFFFFF&"
COLOR_AI = "&H00B0B0B0&"


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print("CMD FAILED:", " ".join(cmd[:8]), file=sys.stderr)
        print(r.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"命令失败 rc={r.returncode}")
    return r


def probe_duration(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def wav_info(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate()), w.getframerate(), w.getnchannels()


def measure_rms_db(path, is_wav=False):
    """实测音频 RMS(dBFS) —— 用确定性增益代替 loudnorm（短音频下 loudnorm 会过度衰减）"""
    import numpy as np
    if is_wav:
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            raw = w.readframes(w.getnframes())
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
    else:
        sr = 16000
        r = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-f", "f32le",
                            "-ac", "1", "-ar", str(sr), "-"],
                           capture_output=True, check=True)
        data = np.frombuffer(r.stdout, dtype=np.float32)
    if data.size == 0:
        return -60.0
    rms = float(np.sqrt(np.mean(data ** 2)))
    return 20.0 * np.log10(max(rms, 1e-9))


def gain_to_target(measured_db, target_db):
    """把实测 RMS 调整到目标 RMS，返回 dB 增益（带上下限）"""
    g = target_db - measured_db
    return max(-24.0, min(24.0, g))


def voice_speech_start(path):
    """
    检测配音的真实语音起点（秒）—— TTS 会在句首留静音/换气（实测 s04 0.4s、s10 0.8s）。
    铁律：字幕与增益必须由真实发音起点驱动，不能按整段起始估算。
    返回 (语音起点秒, 语音段RMS dBFS)
    """
    import numpy as np
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    # 10ms 窗 RMS 包络
    win = max(1, int(sr * 0.01))
    n = len(data) // win
    if n == 0:
        return 0.0, -60.0
    env = np.array([float(np.sqrt(np.mean(data[i * win:(i + 1) * win] ** 2)))
                     for i in range(n)])
    peak = float(env.max())
    if peak <= 1e-6:
        return 0.0, -60.0
    # 阈值 = 峰值的 8%（-22dB），并要求连续 3 窗（30ms）以上才算语音
    thr = peak * 0.08
    start_idx = None
    for i in range(n):
        if env[i] >= thr and i + 2 < n and env[i + 1] >= thr and env[i + 2] >= thr:
            start_idx = i
            break
    if start_idx is None:
        start_idx = int(np.argmax(env))
    # 语音段：从起点到末尾，取有效能量部分
    speech = data[start_idx * win:]
    rms = float(np.sqrt(np.mean(speech ** 2))) if speech.size else 1e-9
    return start_idx * win / float(sr), 20.0 * np.log10(max(rms, 1e-9))


def esc_ass(s):
    s = s.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return s.replace("\n", "\\N")


def build_ass(timings, total_dur, ai_label="本片由AI生成"):
    lines = ["[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}",
             "WrapStyle: 0", "ScaledBorderAndShadow: yes", "YCbCr Matrix: TV.709", "",
             "[V4+ Styles]",
             "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
             "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
             "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
             "MarginR, MarginV, Encoding"]
    lines.append(f"Style: Dialogue,Source Han Sans SC,30,{COLOR_TEXT},{COLOR_TEXT},"
                 f"&H00101010&,&H80000000&,0,0,0,0,100,100,0,0,1,2.4,1,2,50,50,90,1")
    lines.append(f"Style: AILabel,Source Han Sans SC,17,{COLOR_AI},{COLOR_AI},"
                 f"&H00101010&,&H80000000&,0,0,0,0,100,100,0,0,1,1.6,0,2,30,30,42,1")
    lines += ["", "[Events]",
              "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]

    def ts(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    for shot, speaker, text, st, en in timings:
        sp_col = COLOR_SPEAKER.get(speaker, "&H00FFFFFF&")
        body = f"{{\\c{sp_col}}}{esc_ass(speaker)}{{\\c{COLOR_TEXT}}}：{esc_ass(text)}"
        lines.append(f"Dialogue: 0,{ts(st)},{ts(en)},Dialogue,,0,0,0,,{body}")
    lines.append(f"Dialogue: 0,{ts(0)},{ts(total_dur)},AILabel,,0,0,0,,{ai_label}")
    return "\n".join(lines) + "\n"


def render_intro_card(path, title="决裂", subtitle="都市商战悬疑 · 第一话", dur=INTRO_DUR):
    vf = (
        f"drawbox=x=0:y=0:w={W}:h={H}:color=black@1.0:t=fill,"
        f"fade=t=out:st={max(dur-0.3,0):.2f}:d=0.3,"
        f"drawtext=fontfile='{FONT}':text='{title}':fontcolor=white:fontsize=72:"
        f"x=(w-text_w)/2:y=(h-text_h)/2-46:enable='between(t,0.15,{dur-0.15})',"
        f"drawtext=fontfile='{FONT}':text='{subtitle}':fontcolor=0xB0B0B0:fontsize=24:"
        f"x=(w-text_w)/2:y=(h-text_h)/2+56:enable='between(t,0.35,{dur-0.15})'"
    )
    run([FFMPEG, "-y", "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={FPS}:d={dur}",
         "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-t", str(dur), path])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion-dir", default=f"{BASE}/motion_h3_fln")
    ap.add_argument("--out", default=f"{OUT_DIR}/drama_1min_h3.mp4")
    args = ap.parse_args()
    motion_dir = args.motion_dir

    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR)

    seg_paths = []
    timings = []
    video_timeline = INTRO_DUR
    cursor = 0.0
    timeline_rows = []

    print("=== 1. 逐镜合成（环境音 + 配音 duck 混合）===")
    for s in SHOTS:
        name = s["name"]
        vpath = f"{motion_dir}/{name}.mp4"
        if not os.path.exists(vpath):
            print(f"  缺视频 {vpath}，跳过")
            continue
        vdur = probe_duration(vpath)

        apath, speaker, text = None, None, None
        for sp, tx in s["dialogue"]:
            cand = f"{VOICE_DIR}/{name}_{sp}.wav"
            if os.path.exists(cand):
                apath, speaker, text = cand, sp, tx
                break

        is_last = (s["id"] == len(SHOTS))
        seg_dur = LAST_SHOT_DUR if is_last else SHOT_DUR
        seg_frames = LAST_SHOT_FRAMES if is_last else SHOT_FRAMES

        vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
              f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
              f"trim=duration={seg_dur:.4f},setpts=PTS-STARTPTS")

        seg = f"{WORK_DIR}/seg_{name}.mp4"
        if apath is None:
            # 无配音：保留原环境音
            a_rms_nv = measure_rms_db(vpath, is_wav=False)
            a_gain_nv = gain_to_target(a_rms_nv, AMBIENT_TARGET_LUFS)
            run([FFMPEG, "-y", "-i", vpath,
                 "-filter_complex",
                 f"[0:v]{vf}[v];"
                 f"[0:a]aformat=sample_rates={SR}:channel_layouts=stereo,"
                 f"volume={a_gain_nv:.2f}dB,"
                 f"apad=whole_dur={seg_dur:.4f},atrim=duration={seg_dur:.4f}[a]",
                 "-map", "[v]", "-map", "[a]",
                 "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                 "-pix_fmt", "yuv420p", "-r", str(FPS),
                 "-c:a", "aac", "-b:a", "160k", "-ar", str(SR),
                 "-t", f"{seg_dur:.4f}", seg])
            seg_paths.append(seg)
            video_timeline += seg_dur
            cursor += seg_dur
            print(f"  [{name}] 无配音 -> 纯环境音 {seg_dur:.3f}s")
            continue

        adur, asr, ach = wav_info(apath)
        seg_start = video_timeline
        # 真实语音起点（TTS 句首常有静音/换气，s04 0.4s、s10 0.8s）
        speech_off, v_rms = voice_speech_start(apath)
        # 字幕起点 = 镜起点 + 前置留白 + 语音在音频内的偏移（铁律：字幕由真实发音驱动）
        delay_ms = int(round((PAD_HEAD + speech_off) * 1000))
        sub_start = seg_start + PAD_HEAD + speech_off
        sub_end = sub_start + max(adur - speech_off, 0.1)

        # 确定性增益：实测语音段 RMS -> 目标 RMS
        v_gain = gain_to_target(v_rms, VOICE_TARGET_RMS)
        a_rms = measure_rms_db(vpath, is_wav=False)
        a_gain = gain_to_target(a_rms, AMBIENT_TARGET_LUFS)
        print(f"    语音起点 +{speech_off:.2f}s | 电平: 环境音 {a_rms:.1f}dB -> "
              f"{a_gain:+.1f}dB | 配音 {v_rms:.1f}dB -> {v_gain:+.1f}dB", flush=True)

        run([FFMPEG, "-y", "-i", vpath, "-i", apath,
             "-filter_complex",
             f"[0:v]{vf}[v];"
             # 环境音：实测 RMS 归一 + 垫底增益（H3 各镜原生音轨跨度 14.6 LUFS）
             f"[0:a]aformat=sample_rates={SR}:channel_layouts=stereo,"
             f"volume={a_gain:.2f}dB,volume={AMBIENT_GAIN}[amb0];"
             # 配音：延迟 + 实测 RMS 归一（人声清晰可闻）
             f"[1:a]aformat=sample_rates={SR}:channel_layouts=stereo,"
             f"adelay={delay_ms}|{delay_ms},volume={v_gain:.2f}dB[vo0];"
             # sidechain duck：配音触发时环境音压低，保证人声优先
             f"[amb0][vo0]sidechaincompress=threshold=0.02:ratio=10:attack=15:"
             f"release=400:makeup=1[amb];"
             f"[vo0]apad=whole_dur={seg_dur:.4f},atrim=duration={seg_dur:.4f}[vo];"
             f"[amb]apad=whole_dur={seg_dur:.4f},atrim=duration={seg_dur:.4f}[ambf];"
             f"[ambf][vo]amix=inputs=2:duration=longest:dropout_transition=0:"
             f"normalize=0[a]",
             "-map", "[v]", "-map", "[a]",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", "-r", str(FPS),
             "-c:a", "aac", "-b:a", "160k", "-ar", str(SR),
             "-t", f"{seg_dur:.4f}", seg])

        seg_paths.append(seg)
        timings.append((name, speaker, text, sub_start, sub_end))
        timeline_rows.append({
            "shot": name, "speaker": speaker, "text": text,
            "start": round(seg_start, 3), "end": round(seg_start + seg_dur, 3),
            "sub_start": round(sub_start, 3), "sub_end": round(sub_end, 3),
            "voice_dur": round(adur, 3), "video_dur": round(vdur, 3),
        })
        video_timeline += seg_dur
        cursor += seg_dur
        print(f"  [{name}] 视频{vdur:.2f}s 配音{adur:.2f}s -> 片段{seg_dur:.3f}s "
              f"字幕 {sub_start:.2f}~{sub_end:.2f}")

    if not seg_paths:
        raise SystemExit("没有可用的视频片段")

    print("\n=== 2. 拼接片段 ===")
    list_txt = f"{WORK_DIR}/concat.txt"
    with open(list_txt, "w") as f:
        for p in seg_paths:
            f.write(f"file '{p}'\n")
    concat_main = f"{WORK_DIR}/concat_main.mp4"
    run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", list_txt,
         "-filter_complex",
         f"[0:v]fps={FPS},format=yuv420p,setpts=PTS-STARTPTS[v];"
         f"[0:a]aresample={SR},aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS[a]",
         "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "160k", "-ar", str(SR),
         concat_main])

    print("\n=== 3. 片头卡 ===")
    intro = f"{WORK_DIR}/intro.mp4"
    render_intro_card(intro)
    main_with_intro = f"{WORK_DIR}/main_with_intro.mkv"
    run([FFMPEG, "-y", "-i", intro, "-i", concat_main,
         "-filter_complex",
         f"[0:v]fps={FPS},format=yuv420p,setpts=PTS-STARTPTS[v0];"
         f"[1:v]fps={FPS},format=yuv420p,setpts=PTS-STARTPTS[v1];"
         f"[v0][v1]concat=n=2:v=1:a=0[v];"
         f"anullsrc=channel_layout=stereo:sample_rate={SR}:d={INTRO_DUR}[sil];"
         f"[1:a]aresample={SR},aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS[a1];"
         f"[sil][a1]concat=n=2:v=0:a=1[a]",
         "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "160k", "-ar", str(SR), main_with_intro])

    print("\n=== 4. 字幕 ===")
    total_dur = INTRO_DUR + cursor
    ass_path = f"{WORK_DIR}/subs.ass"
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(build_ass(timings, total_dur))
    print(f"  字幕 {len(timings)} 条，总时长 {total_dur:.3f}s")

    print("\n=== 5. 最终输出 ===")
    fade = f"afade=t=out:st={max(total_dur-OUTRO_FADE,0):.3f}:d={OUTRO_FADE}"
    cmd = [FFMPEG, "-y", "-i", main_with_intro,
           "-filter_complex",
           # 只做真峰限制（保证 <= -1.5 dBTP），不再做 loudnorm 二次压缩，
           # 否则会把已归一化的人声再压低（实测导致 ASR 识别不出台词）
           f"[0:a]{fade},alimiter=limit=0.841:level=disabled:attack=5:release=50[aout]",
           "-map", "0:v", "-map", "[aout]",
           "-vf",
           "eq=gamma=1.12:brightness=0.012:contrast=1.05:saturation=1.06,"
           "curves=all='0/0.02 0.25/0.30 1/1',"
           f"ass='{ass_path}'",
           "-c:v", "libx264", "-preset", "slow", "-crf", "19",
           "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-c:a", "aac", "-b:a", "192k", "-ar", str(SR),
           "-t", f"{TARGET_DUR:.3f}",
           "-movflags", "+faststart", args.out]
    run(cmd)

    print(f"\n成片: {args.out}")
    print(f"总时长 {total_dur:.3f}s")
    with open(f"{OUT_DIR}/timeline_h3.json", "w", encoding="utf-8") as f:
        json.dump({"total_dur": round(total_dur, 3), "fps": FPS,
                   "intro_dur": INTRO_DUR, "shots": timeline_rows},
                  f, ensure_ascii=False, indent=2)
    print(f"时间轴: {OUT_DIR}/timeline_h3.json")


if __name__ == "__main__":
    main()
