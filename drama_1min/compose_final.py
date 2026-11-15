#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《决裂》1分钟短剧 — 终合成

输入：motion/*.mp4（12段 @16fps）+ voice/*.wav（12句配音）
输出：final/drama_1min_final.mp4（480x832 / 16fps / 约60s）

流程：
  1. 逐镜：视频轨 + 配音轨对齐（配音居中，前后留静音垫）
  2. ASS 字幕：分色（说话人角色名用主色，台词用白色），字形用 SourceHanSansSC-Bold.otf
  3. 片头卡（2.0s）：剧名 + 副标题
  4. 全片 AI 生成标识（右下角常驻淡入）
  5. BGM（若提供）低音量垫底 + 片尾淡出

音画同步铁律：
  - 配音起点 = 该镜起点 + PAD_HEAD（0.35s）
  - 字幕时间轴由真实配音时长驱动（读 wav 帧数），禁按字数估算
  - 句间静音垫 >= 0.45s

用法：compose_final.py [--bgm 路径] [--out 输出路径]
"""
import os, sys, json, wave, subprocess, shutil, argparse
from math import gcd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shots_1min import SHOTS

BASE = "/ssd2/duanju/drama_1min"
MOTION_DIR = f"{BASE}/motion"
VOICE_DIR = f"{BASE}/voice"
OUT_DIR = f"{BASE}/final"
WORK_DIR = f"{BASE}/compose_work"
FONT = "/ssd2/short_drama/fonts/SourceHanSansSC-Bold.otf"

FFMPEG = "/usr/local/bin/ffmpeg"
FFPROBE = "/usr/local/bin/ffprobe"

FPS = 16
W, H = 480, 832
SR = 44100
SHOT_DUR = 5.0         # 每镜 80 帧 @16fps；12 镜 = 60.0s 精确
LAST_SHOT_DUR = 4.0    # 末镜收短 1.0s，抵消片头 1.0s => 成片 60.0s

PAD_HEAD = 0.35        # 镜内配音前置留白
PAD_TAIL = 0.45        # 句间静音垫（铁律 >= 0.45s）
INTRO_DUR = 1.0        # 片头卡（正片 12x5.0s + 片头 1.0s = 61.0s；末镜再收 1.0s -> 60.0s）
OUTRO_FADE = 1.0       # 片尾淡出

# 角色分色（ASS &HBBGGRR）
COLOR_SPEAKER = {
    "苏清晚": "&H00E8C4FF&",   # 淡紫
    "苏蔓":   "&H0080FFB0&",   # 青绿
    "宋佳宁": "&H00C8D2FF&",   # 淡蓝
    "裴九渊": "&H004060C0&",   # 暗红
    "陆沉":   "&H00D8D0C0&",   # 暖灰金
}
COLOR_TEXT = "&H00FFFFFF&"    # 白
COLOR_AI = "&H00B0B0B0&"       # 灰


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print("CMD FAILED:", " ".join(cmd[:6]), file=sys.stderr)
        print(r.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"命令失败 rc={r.returncode}")
    return r


def probe_duration(path):
    r = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", path],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def wav_info(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate()), w.getframerate(), w.getnchannels()


def esc_ass(s):
    """ASS 转义：花括号、换行、反斜杠"""
    s = s.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return s.replace("\n", "\\N")


def build_ass(timings, total_dur, ai_label="本片由AI生成"):
    """
    timings: [(shot_name, speaker, text, start_sec, end_sec)]
    片头卡单独用 drawtext/PIL 渲染更可控，这里 ASS 只负责对白字幕 + AI 标识。
    """
    lines = []
    lines.append("[Script Info]")
    lines.append("ScriptType: v4.00+")
    lines.append(f"PlayResX: {W}")
    lines.append(f"PlayResY: {H}")
    lines.append("WrapStyle: 0")
    lines.append("ScaledBorderAndShadow: yes")
    lines.append("YCbCr Matrix: TV.709")
    lines.append("")
    lines.append("[V4+ Styles]")
    lines.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                 "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
                 "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
                 "MarginR, MarginV, Encoding")
    # 对白样式：底部居中，MarginV=90
    lines.append(f"Style: Dialogue,Source Han Sans SC,30,{COLOR_TEXT},{COLOR_TEXT},"
                 f"&H00101010&,&H80000000&,0,0,0,0,100,100,0,0,1,2.4,1,2,50,50,90,1")
    # AI 标识：右下角
    lines.append(f"Style: AILabel,Source Han Sans SC,17,{COLOR_AI},{COLOR_AI},"
                 f"&H00101010&,&H80000000&,0,0,0,0,100,100,0,0,1,1.6,0,2,30,30,42,1")
    lines.append("")
    lines.append("[Events]")
    lines.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")

    def ts(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    for shot, speaker, text, st, en in timings:
        sp_col = COLOR_SPEAKER.get(speaker, "&H00FFFFFF&")
        # 角色名 + 台词 同一行，角色名分色
        body = f"{{\\c{sp_col}}}{esc_ass(speaker)}{{\\c{COLOR_TEXT}}}：{esc_ass(text)}"
        lines.append(f"Dialogue: 0,{ts(st)},{ts(en)},Dialogue,,0,0,0,,{body}")

    # AI 标识：全程常驻（淡入淡出）
    lines.append(f"Dialogue: 0,{ts(0)},{ts(total_dur)},AILabel,,0,0,0,,{ai_label}")
    return "\n".join(lines) + "\n"


def render_intro_card(path, title="决裂", subtitle="都市商战悬疑 · 第一话", dur=INTRO_DUR):
    """用 ffmpeg drawtext 渲染片头卡（黑底 + 标题）"""
    # 标题在 0.15s 后出现，副标题稍晚，片尾淡出
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
    ap.add_argument("--bgm", default="")
    ap.add_argument("--out", default=f"{OUT_DIR}/drama_1min_final.mp4")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
    os.makedirs(WORK_DIR)

    # ---- 1. 逐镜构建 视频+音频，并记录时间轴 ----
    seg_paths = []
    timings = []
    cursor = 0.0  # 已用视频时间轴（片头之后）
    INTRO = INTRO_DUR
    video_timeline = INTRO

    print("=== 1. 逐镜合成 ===")
    for s in SHOTS:
        name = s["name"]
        vpath = f"{MOTION_DIR}/{name}.mp4"
        if not os.path.exists(vpath):
            print(f"  缺视频 {vpath}，跳过")
            continue
        vdur = probe_duration(vpath)

        # 配音
        apath = None
        for sp, tx in s["dialogue"]:
            cand = f"{VOICE_DIR}/{name}_{sp}.wav"
            if os.path.exists(cand):
                apath = cand
                speaker, text = sp, tx
                break
        if apath is None:
            print(f"  [{name}] 无配音 -> 纯画面")
            seg = f"{WORK_DIR}/seg_{name}.mp4"
            seg_dur = LAST_SHOT_DUR if s["id"] == len(SHOTS) else SHOT_DUR
            vfix = (f"tpad=stop_mode=clone:stop_duration={seg_dur - vdur:.3f},"
                    if seg_dur > vdur else "")
            run([FFMPEG, "-y", "-i", vpath, "-f", "lavfi", "-i",
                 f"anullsrc=channel_layout=stereo:sample_rate={SR}",
                 "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                        f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
                        f"{vfix}"
                        f"trim=duration={seg_dur:.3f},setpts=PTS-STARTPTS",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
                 "-c:a", "aac", "-ar", str(SR),
                 "-af", f"apad=whole_dur={seg_dur:.3f},atrim=duration={seg_dur:.3f}",
                 "-t", f"{seg_dur:.3f}", seg])
            seg_paths.append(seg)
            cursor += seg_dur
            video_timeline += seg_dur
            continue

        adur, asr, ach = wav_info(apath)
        # 配音起点 = 镜内 PAD_HEAD；字幕同步
        seg_start = video_timeline
        sub_start = seg_start + PAD_HEAD
        sub_end = sub_start + adur

        # 目标片段时长 = 配音 + 前垫 + 后垫；且不短于 SHOT_DUR（12镜 x 5.0s = 精确 60.0s）
        # 注：VHS_VideoCombine 实测每段只落 77 帧（4.8125s），此处用 tpad 补足尾帧
        seg_dur = max(SHOT_DUR, PAD_HEAD + adur + PAD_TAIL)
        # 末镜收 1.0s，配合 1.0s 片头 => 总长精确 60.0s
        if s["id"] == len(SHOTS):
            seg_dur = max(LAST_SHOT_DUR, PAD_HEAD + adur + PAD_TAIL)
        # 视频对齐到 seg_dur：长则裁剪，短则末帧定格补足；音频用 adelay 精确对齐
        delay_ms = int(round(PAD_HEAD * 1000))
        if seg_dur > vdur:
            vfix = f"tpad=stop_mode=clone:stop_duration={seg_dur - vdur:.3f},"
        else:
            vfix = ""   # 目标更短 -> 直接 trim 裁掉尾巴
        vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
              f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
              f"{vfix}"
              f"trim=duration={seg_dur:.3f},setpts=PTS-STARTPTS")
        seg = f"{WORK_DIR}/seg_{name}.mp4"
        # 关键：apad 把音频补静音到 seg_dur，否则 concat -c copy 会按视频时长拉伸音频，
        # 导致成片语音时间轴逐镜前移（实测 12 镜累计偏移 -17s）
        run([FFMPEG, "-y", "-i", vpath, "-i", apath,
             "-filter_complex",
             f"[0:v]{vf}[v];"
             f"[1:a]aformat=sample_rates={SR}:channel_layouts=stereo,"
             f"adelay={delay_ms}|{delay_ms},apad=whole_dur={seg_dur:.3f},"
             f"atrim=duration={seg_dur:.3f}[a]",
             "-map", "[v]", "-map", "[a]",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", "-r", str(FPS),
             "-c:a", "aac", "-b:a", "128k", "-ar", str(SR),
             "-t", f"{seg_dur:.3f}", seg])

        seg_paths.append(seg)
        timings.append((name, speaker, text, sub_start, sub_end))
        cursor += seg_dur
        video_timeline += seg_dur
        print(f"  [{name}] 视频{vdur:.2f}s 配音{adur:.2f}s -> 片段{seg_dur:.2f}s "
              f"字幕 {sub_start:.2f}~{sub_end:.2f}")

    if not seg_paths:
        raise SystemExit("没有可用的视频片段")

    # ---- 2. 拼接 ----
    print("\n=== 2. 拼接片段 ===")
    list_txt = f"{WORK_DIR}/concat.txt"
    with open(list_txt, "w") as f:
        for p in seg_paths:
            f.write(f"file '{p}'\n")
    concat_no_intro = f"{WORK_DIR}/concat_main.mp4"
    # 用 filter concat 重编码（不用 -c copy）：
    # -c copy 依赖各段时间戳基线一致，实测会出现音频轨比视频轨短（57.53s vs 59.0s），
    # 后续拼片头时音画错位。filter concat 会重新对时基。
    run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", list_txt,
         "-filter_complex",
         "[0:v]fps=16,format=yuv420p,setpts=PTS-STARTPTS[v];"
         "[0:a]aresample=44100,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS[a]",
         "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "128k", "-ar", str(SR),
         concat_no_intro])

    # ---- 3. 片头卡 ----
    print("\n=== 3. 片头卡 ===")
    intro = f"{WORK_DIR}/intro.mp4"
    render_intro_card(intro)
    main_with_intro = f"{WORK_DIR}/main_with_intro.mkv"
    # 视频与音频一次性 concat（片头静音 + 正片），保证两条轨严格同时起算
    run([FFMPEG, "-y", "-i", intro, "-i", concat_no_intro,
         "-filter_complex",
         "[0:v]fps=16,format=yuv420p,setpts=PTS-STARTPTS[v0];"
         "[1:v]fps=16,format=yuv420p,setpts=PTS-STARTPTS[v1];"
         "[v0][v1]concat=n=2:v=1:a=0[v];"
         f"anullsrc=channel_layout=stereo:sample_rate={SR}:d={INTRO}[sil];"
         "[1:a]aresample=44100,aformat=channel_layouts=stereo,asetpts=PTS-STARTPTS[a1];"
         "[sil][a1]concat=n=2:v=0:a=1[a]",
         "-map", "[v]", "-map", "[a]",
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-c:a", "aac", "-b:a", "128k", "-ar", str(SR), main_with_intro])

    # ---- 4. 字幕 ----
    print("\n=== 4. 字幕 ===")
    total_dur = INTRO + cursor
    ass_path = f"{WORK_DIR}/subs.ass"
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(build_ass(timings, total_dur))
    print(f"  字幕 {len(timings)} 条，总时长 {total_dur:.2f}s")

    # ---- 5. BGM ----
    bgm_args = []
    if args.bgm and os.path.exists(args.bgm):
        bgm_args = ["-stream_loop", "-1", "-i", args.bgm]
        print(f"  BGM: {args.bgm}")

    # ---- 6. 最终输出 ----
    print("\n=== 5. 最终输出 ===")
    cmd = [FFMPEG, "-y", "-i", main_with_intro]
    cmd += bgm_args
    fade = f"afade=t=out:st={max(total_dur-OUTRO_FADE,0):.3f}:d={OUTRO_FADE}"
    if bgm_args:
        cmd += ["-filter_complex",
                f"[0:a]volume=1.0,{fade}[a1];"
                f"[1:a]volume=0.12,afade=t=in:d=1.5,"
                f"afade=t=out:st={max(total_dur-2,0):.3f}:d=2[a2];"
                f"[a1][a2]amix=inputs=2:duration=first:dropout_transition=0[aout]",
                "-map", "0:v", "-map", "[aout]"]
    else:
        cmd += ["-filter_complex", f"[0:a]{fade}[aout]",
                "-map", "0:v", "-map", "[aout]"]
    cmd += ["-vf",
            # 轻度提亮 + 暗部提亮：商战夜景底图整体偏暗（实测全片亮度均值 0.184，
            # s10 人物区仅 0.138、s09 背景 57.5% 死黑），做 gamma/eq 提升可读性
            "eq=gamma=1.18:brightness=0.015:contrast=1.06:saturation=1.04,"
            "curves=all='0/0.02 0.25/0.30 1/1',"
            f"ass='{ass_path}'",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", str(FPS), "-c:a", "aac", "-b:a", "160k",
            "-ar", str(SR), "-t", f"{total_dur:.3f}",
            "-movflags", "+faststart", args.out]
    run(cmd)

    final_dur = probe_duration(args.out)
    print(f"\n✅ 合成完成: {args.out}")
    print(f"   时长 {final_dur:.2f}s (计划 {total_dur:.2f}s, 差 {abs(final_dur-total_dur):.2f}s)")

    # 写时间轴报告
    with open(f"{OUT_DIR}/timeline.json", "w", encoding="utf-8") as f:
        json.dump({"total_dur": round(total_dur, 3), "final_dur": round(final_dur, 3),
                   "intro_dur": INTRO, "pad_head": PAD_HEAD, "pad_tail": PAD_TAIL,
                   "shots": [{"shot": t[0], "speaker": t[1], "text": t[2],
                              "start": round(t[3], 3), "end": round(t[4], 3)}
                             for t in timings]}, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
