#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《决裂》H3 版终合成 v2 —— 音频全 Python 样本级处理，FFmpeg 只做视频与封装。

与 v1 的差别（v1 的 ffmpeg 滤镜链有 3 个致命缺陷，见 mix_audio.py 顶部说明）：
  v1: [vo0] 双接 sidechaincompress + apad → 配音丢失/错位；
      adelay 推迟整段 wav（含句首静音）→ 字幕早于发音 0.26~0.90s；
      adelay 后接 apad/atrim 时基被推歪 → 台词全部滑到镜尾
  v2: 配音在 numpy 里按「真实发声点」精确放置（样本级）；
      字幕时间轴 = 发声点时间（零误差）；环境音 duck 用 numpy 包络曲线
      整片 60s 混音后统一做两遍 loudnorm（长信号，不会过度衰减）

用法：compose_h3_v2.py [--motion-dir motion_h3_fln] [--out final/drama_1min_h3.mp4]
"""
import os
import sys
import json
import shutil
import subprocess
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shots_1min import SHOTS
import mix_audio

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

SHOT_FRAMES = 118            # 常规镜 118 帧 @24fps = 4.9167s
SHOT_DUR = SHOT_FRAMES / FPS
LAST_SHOT_FRAMES = 124       # 末镜完整 124 帧 = 5.1667s
LAST_SHOT_DUR = LAST_SHOT_FRAMES / FPS
INTRO_DUR = 1.0              # 片头卡
TARGET_DUR = 60.0            # 成片精确时长
PAD_HEAD = 0.30              # 镜内配音前置留白（发音点落点）

TARGET_LUFS = -16.0          # GY/T 377-2023：-15±2 LKFS
TARGET_TP = -1.5             # 真峰值 <= -1 dBTP

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
        print("CMD FAILED:", " ".join(cmd[:10]), file=sys.stderr)
        print(r.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"命令失败 rc={r.returncode}")
    return r


def esc_ass(s):
    s = s.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
    return s.replace("\n", "\\N")


def build_ass(rows, total_dur, ai_label="本片由AI生成"):
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

    n = 0
    for r in rows:
        if r.get("sub_start") is None:
            continue
        sp_col = COLOR_SPEAKER.get(r.get("speaker"), "&H00FFFFFF&")
        body = f"{{\\c{sp_col}}}{esc_ass(r['speaker'])}{{\\c{COLOR_TEXT}}}：{esc_ass(r['text'])}"
        lines.append(f"Dialogue: 0,{ts(r['sub_start'])},{ts(r['sub_end'])},"
                     f"Dialogue,,0,0,0,,{body}")
        n += 1
    lines.append(f"Dialogue: 0,{ts(0)},{ts(total_dur)},AILabel,,0,0,0,,{ai_label}")
    return "\n".join(lines) + "\n", n


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


def two_pass_loudnorm(src_wav, dst_wav):
    """整片 60s 混音 → 两遍 loudnorm（长信号，安全）"""
    r = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", src_wav,
                        "-af", f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA=11:"
                               f"print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    txt = r.stderr
    i = txt.rfind("{")
    j = txt.rfind("}")
    if i < 0 or j < 0:
        print("  loudnorm 测量失败，退回单遍", file=sys.stderr)
        run([FFMPEG, "-y", "-i", src_wav,
             "-af", f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA=11",
             "-ar", str(SR), "-c:a", "pcm_s16le", dst_wav])
        return {}
    j = json.loads(txt[i:j + 1])
    measured = j.get("input_i")
    if measured in (None, "-inf"):
        run([FFMPEG, "-y", "-i", src_wav,
             "-af", f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA=11",
             "-ar", str(SR), "-c:a", "pcm_s16le", dst_wav])
        return j
    print(f"  loudnorm 输入 {measured} LUFS -> 目标 {TARGET_LUFS} LUFS")
    # 第二遍用 measured_* 做线性修正（linear=true 保持动态不压缩）
    run([FFMPEG, "-y", "-i", src_wav,
         "-af", (f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA=11:"
                 f"measured_I={measured}:measured_TP={j.get('input_tp')}:"
                 f"measured_LRA={j.get('input_lra')}:"
                 f"measured_thresh={j.get('input_thresh')}:"
                 f"offset={j.get('target_offset')}:linear=true:print_format=summary"),
         "-ar", str(SR), "-c:a", "pcm_s16le", dst_wav])
    return j


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

    # ---------- 1. 组装镜头表 ----------
    shots = []
    for s in SHOTS:
        name = s["name"]
        vpath = f"{motion_dir}/{name}.mp4"
        if not os.path.exists(vpath):
            print(f"  缺视频 {vpath}，跳过")
            continue
        voice_path = speaker = text = None
        for sp, tx in s["dialogue"]:
            cand = f"{VOICE_DIR}/{name}_{sp}.wav"
            if os.path.exists(cand):
                voice_path, speaker, text = cand, sp, tx
                break
        is_last = (s["id"] == len(SHOTS))
        shots.append({
            "id": s["id"], "name": name,
            "seg_dur": LAST_SHOT_DUR if is_last else SHOT_DUR,
            "voice_path": voice_path, "speaker": speaker, "text": text,
        })
    if not shots:
        raise SystemExit("没有可用镜头")

    # ---------- 2. 视频轨：逐镜裁剪 + 片头 + 拼接 ----------
    print("=== 1. 视频轨（无声）===")
    seg_paths = []
    for s in shots:
        seg = f"{WORK_DIR}/v_{s['name']}.mp4"
        vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
              f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
              f"trim=duration={s['seg_dur']:.4f},setpts=PTS-STARTPTS")
        run([FFMPEG, "-y", "-i", f"{motion_dir}/{s['name']}.mp4",
             "-vf", vf, "-an",
             "-c:v", "libx264", "-preset", "medium", "-crf", "20",
             "-pix_fmt", "yuv420p", "-r", str(FPS),
             "-t", f"{s['seg_dur']:.4f}", seg])
        seg_paths.append(seg)
        print(f"  [{s['name']}] {s['seg_dur']:.3f}s")

    intro = f"{WORK_DIR}/intro.mp4"
    render_intro_card(intro)
    # 用 concat 滤镜而非 concat demuxer —— demuxer 实测丢 20 帧（1426/1446），
    # 会导致字幕与画面逐渐漂移；concat 滤镜帧级精确（实测 1446/1446）
    all_segs = [intro] + seg_paths
    cmd = [FFMPEG, "-y"]
    for p in all_segs:
        cmd += ["-i", p]
    fc = []
    for i in range(len(all_segs)):
        fc.append(f"[{i}:v]fps={FPS},format=yuv420p,setpts=PTS-STARTPTS[v{i}]")
    chain = "".join(f"[v{i}]" for i in range(len(all_segs)))
    fc.append(f"{chain}concat=n={len(all_segs)}:v=1:a=0[vout]")
    video_only = f"{WORK_DIR}/video_only.mp4"
    cmd += ["-filter_complex", ";".join(fc), "-map", "[vout]",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", str(FPS), video_only]
    run(cmd)
    print(f"  视频轨: {video_only}")

    # ---------- 3. 音频轨：Python 样本级混合 ----------
    print("\n=== 2. 音频混音（Python 样本级）===")
    raw_mix = f"{WORK_DIR}/mix_raw.wav"
    norm_mix = f"{WORK_DIR}/mix_norm.wav"
    timeline_path = f"{OUT_DIR}/timeline_h3.json"
    rows = mix_audio.build_full_mix(
        shots, motion_dir, VOICE_DIR, INTRO_DUR, raw_mix, timeline_path)

    print("\n=== 3. 整片响度归一（两遍 loudnorm）===")
    ln = two_pass_loudnorm(raw_mix, norm_mix)

    # ---------- 4. 字幕 ----------
    total_dur = INTRO_DUR + sum(s["seg_dur"] for s in shots)
    ass_path = f"{WORK_DIR}/subs.ass"
    ass_txt, n_sub = build_ass(rows, total_dur)
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_txt)
    print(f"\n=== 4. 字幕 {n_sub} 条，总时长 {total_dur:.3f}s ===")

    # ---------- 5. 最终封装 ----------
    print("=== 5. 最终输出 ===")
    fade = f"afade=t=out:st={max(total_dur-1.0,0):.3f}:d=1.0"
    cmd = [FFMPEG, "-y", "-i", video_only, "-i", norm_mix,
           "-filter_complex",
           f"[1:a]{fade},alimiter=limit=0.841:level=disabled:attack=5:release=50[aout]",
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
    print(f"总时长 {total_dur:.3f}s（输出裁到 {TARGET_DUR:.3f}s）")
    if ln:
        print(f"响度: 输入 {ln.get('input_i')} LUFS -> 目标 {TARGET_LUFS} LUFS "
              f"(TP {ln.get('input_tp')} dBTP)")
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump({"total_dur": round(total_dur, 3), "fps": FPS,
                   "intro_dur": INTRO_DUR, "target_lufs": TARGET_LUFS,
                   "shots": rows}, f, ensure_ascii=False, indent=2)
    print(f"时间轴: {timeline_path}")


if __name__ == "__main__":
    main()
