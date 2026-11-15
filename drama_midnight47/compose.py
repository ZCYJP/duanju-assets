#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《零点四十七单》— 成片合成（两个样本各自成片）

管线（全 Python/numpy 样本级混音，FFmpeg 只做视频与封装）：
  1. 镜级拼接：每镜 N 段（首尾相接的 FL2VA 段）用 concat 滤镜合并 → motion/{shot}.mp4
  2. 视频轨：逐镜裁剪（按帧数，不用秒数——trim=秒数会向上取整到整帧导致累积漂移）
     + 片头卡，用 concat 滤镜拼接（不用 demuxer，demuxer 实测丢帧）
  3. 音频轨：mix_audio.build_full_mix —— 配音按「真实发声点」样本级放置，
     字幕时间轴 = 发声点，零误差
  4. 整片两遍 loudnorm → -14 LUFS / TP -1.5（短剧标准），不接 alimiter（会推高 1.3 LU）
  5. ASS 烧字幕（只留文字+描边，不加全屏蒙版——全屏半透明蒙版会洗白画面）
  6. 封装

用法：compose.py --sample A
      compose.py --sample B
"""
import os
import sys
import json
import shutil
import subprocess
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from story import SHOTS, SAMPLES
import mix_audio

BASE = "/ssd2/duanju/drama_midnight47"
VOICE_DIR = f"{BASE}/voice"
OUT_DIR = f"{BASE}/final"
WORK_DIR = f"{BASE}/compose_work"
SEGS_DIR = f"{BASE}/motion_segs"
MOTION_DIR = f"{BASE}/motion"
FONT = "/ssd2/short_drama/fonts/SourceHanSansSC-Bold.otf"

FFMPEG = "/ssd2/miniconda3/envs/videogen/bin/ffmpeg"

FPS = 24
W, H = 480, 832
SR = 44100

SEG_FRAMES = 124              # H3 FL2VA 每段 124 帧 @24fps = 5.1667s
INTRO_FRAMES = 36             # 片头卡 36 帧 = 1.5s
PAD_HEAD = 0.40               # 镜内配音前置留白（发音点落点）

TARGET_LUFS = -14.0           # 短剧标准 -14 LUFS
TARGET_TP = -1.5

COLOR_SPEAKER = {
    "王大川": "&H00E8C4FF&",
    "邱以沫": "&H00A8E0FF&",
}
COLOR_TEXT = "&H00FFFFFF&"
COLOR_AI = "&H00B0B0B0&"


def run(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print("CMD FAILED:", " ".join(cmd[:12]), file=sys.stderr)
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
    # 只留文字 + 描边 + 轻阴影，不加全屏半透明蒙版（会整体洗白画面）
    lines.append(f"Style: Dialogue,Source Han Sans SC,32,{COLOR_TEXT},{COLOR_TEXT},"
                 f"&H00101010&,&H80000000&,0,0,0,0,100,100,0,0,1,2.6,1,2,42,42,190,1")
    lines.append(f"Style: AILabel,Source Han Sans SC,16,{COLOR_AI},{COLOR_AI},"
                 f"&H00101010&,&H80000000&,0,0,0,0,100,100,0,0,1,1.6,0,2,26,26,40,1")
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


def render_intro_card(path, title, subtitle, dur):
    vf = (
        f"drawbox=x=0:y=0:w={W}:h={H}:color=black@1.0:t=fill,"
        f"fade=t=out:st={max(dur-0.3,0):.2f}:d=0.3,"
        f"drawtext=fontfile='{FONT}':text='{title}':fontcolor=white:fontsize=76:"
        f"x=(w-text_w)/2:y=(h-text_h)/2-52:enable='between(t,0.12,{dur-0.12})',"
        f"drawtext=fontfile='{FONT}':text='{subtitle}':fontcolor=0xB8B8B8:fontsize=25:"
        f"x=(w-text_w)/2:y=(h-text_h)/2+58:enable='between(t,0.30,{dur-0.12})'"
    )
    run([FFMPEG, "-y", "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={FPS}:d={dur}",
         "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(FPS),
         "-frames:v", str(INTRO_FRAMES), path])


def concat_segments(shot_name, seg_count):
    """把一镜的 N 段 FL2VA 视频（首尾相接）拼成 motion/{shot}.mp4"""
    os.makedirs(MOTION_DIR, exist_ok=True)
    dst = f"{MOTION_DIR}/{shot_name}.mp4"
    segs = []
    for k in range(seg_count):
        p = f"{SEGS_DIR}/{shot_name}_s{k}.mp4"
        if not os.path.exists(p):
            break
        segs.append(p)
    if not segs:
        return None
    if len(segs) == 1:
        shutil.copy(segs[0], dst)
        return dst
    cmd = [FFMPEG, "-y"]
    for p in segs:
        cmd += ["-i", p]
    fc = []
    for i in range(len(segs)):
        fc.append(f"[{i}:v]fps={FPS},format=yuv420p,setpts=PTS-STARTPTS[v{i}]")
        fc.append(f"[{i}:a]aresample={SR},aformat=sample_fmts=fltp:"
                  f"channel_layouts=stereo,asetpts=PTS-STARTPTS[a{i}]")
    chain = "".join(f"[v{i}][a{i}]" for i in range(len(segs)))
    fc.append(f"{chain}concat=n={len(segs)}:v=1:a=1[vout][aout]")
    cmd += ["-filter_complex", ";".join(fc), "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-pix_fmt", "yuv420p", "-r", str(FPS),
            "-c:a", "aac", "-b:a", "192k", "-ar", str(SR), dst]
    run(cmd)
    return dst


def measure_loudness(path):
    """实测某音频文件的 LUFS / TP"""
    r = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", path,
                        "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    txt = r.stderr
    i = txt.rfind("{")
    j = txt.rfind("}")
    if i < 0 or j < 0:
        return {}
    return json.loads(txt[i:j + 1])


def apply_gain(src_wav, dst_wav, gain_db):
    run([FFMPEG, "-y", "-i", src_wav, "-af", f"volume={gain_db:.3f}dB",
         "-ar", str(SR), "-c:a", "pcm_s16le", dst_wav])


def two_pass_loudnorm(src_wav, dst_wav):
    """
    整片混音 → 峰值预压缩 → loudnorm 两遍 → 归一化（已实测确定的链路）。

    为什么必须先压缩（矩阵实验结论，样本A 原始混音 -23.25 LUFS / TP -0.31）：
      本片峰均比 22.9dB（对话密集短剧，峰值集中在配音起始爆破音）。
      直接 loudnorm 会为守住 TP=-1.5 而放弃提响度 → 卡在 -16.56 LUFS。
      实测各链路结果：
        单遍动态 loudnorm            → -16.56 LUFS / TP -1.48   ✗
        强压缩+限幅（单独）            → -33.43 LUFS / TP -17.38  ✗
        强压缩限幅 → loudnorm（E4）    → -14.43 LUFS / TP -1.50   ✓
      即：先压动态腾出峰值余量，再由 loudnorm 做最终整体归一。
    """
    # 1) 峰值预压缩 + 限幅（把峰均比压下来，给 loudnorm 留出提升空间）
    pre = src_wav.replace(".wav", "_pre.wav")
    run([FFMPEG, "-y", "-i", src_wav,
         "-af", ("acompressor=threshold=-28dB:ratio=6:attack=3:release=150:makeup=1,"
                 "alimiter=limit=0.5:level=disabled:attack=3:release=60"),
         "-ar", str(SR), "-c:a", "pcm_s16le", pre])

    # 2) loudnorm 两遍（第一遍测量）
    r = subprocess.run([FFMPEG, "-hide_banner", "-nostats", "-i", pre,
                        "-af", f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA=11:"
                               f"print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    txt = r.stderr
    i = txt.rfind("{")
    j = txt.rfind("}")
    if i < 0 or j < 0:
        print("  loudnorm 测量失败，退回单遍", file=sys.stderr)
        run([FFMPEG, "-y", "-i", pre,
             "-af", f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA=11",
             "-ar", str(SR), "-c:a", "pcm_s16le", dst_wav])
        os.remove(pre)
        return {}
    j = json.loads(txt[i:j + 1])
    measured = j.get("input_i")
    if measured in (None, "-inf"):
        run([FFMPEG, "-y", "-i", pre,
             "-af", f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA=11",
             "-ar", str(SR), "-c:a", "pcm_s16le", dst_wav])
        os.remove(pre)
        return j
    print(f"  压缩后 {measured} LUFS -> loudnorm 目标 {TARGET_LUFS} LUFS")
    # 第二遍：带 measured 参数做精确归一（动态模式）
    run([FFMPEG, "-y", "-i", pre,
         "-af", (f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA=11:"
                 f"measured_I={measured}:measured_TP={j.get('input_tp')}:"
                 f"measured_LRA={j.get('input_lra')}:"
                 f"measured_thresh={j.get('input_thresh')}:"
                 f"offset={j.get('target_offset')}:print_format=summary"),
         "-ar", str(SR), "-c:a", "pcm_s16le", dst_wav])
    os.remove(pre)
    return j


def normalize_to_target(wav_path, total_dur):
    """
    最终归一微调：fade 烘进 wav → 实测 → 静态增益修正（TP 上限内）。

    压缩与限幅已在 two_pass_loudnorm 的第 1 步完成，
    这里不再重复压缩（重复会过度压动态，实测掉到 -20 LUFS）。
    """
    # 1) 尾音 fade 烘进 wav（先做再测，否则测量不含 fade、结果偏乐观）
    faded = wav_path.replace(".wav", "_faded.wav")
    fade_st = max(total_dur - 1.0, 0.0)
    run([FFMPEG, "-y", "-i", wav_path,
         "-af", f"afade=t=out:st={fade_st:.3f}:d=1.0",
         "-ar", str(SR), "-c:a", "pcm_s16le", faded])

    # 2) 实测 → 静态增益校正（TP 允许范围内）
    m = measure_loudness(faded)
    i0 = m.get("input_i")
    tp0 = m.get("input_tp")
    if i0 in (None, "-inf") or tp0 in (None, "-inf"):
        print("  实测失败，跳过增益校正")
        os.replace(faded, wav_path)
        return m
    need = TARGET_LUFS - float(i0)
    max_gain = TARGET_TP - float(tp0)   # 保证真峰值不超过 -1.5 dBTP
    gain = min(need, max_gain)
    print(f"  实测 {i0} LUFS / TP {tp0} dBTP  →  需增益 {need:+.2f}dB"
          f"（TP 上限允许 {max_gain:+.2f}dB，实际用 {gain:+.2f}dB）")
    if abs(gain) > 0.05:
        apply_gain(faded, wav_path, gain)
        os.remove(faded)
    else:
        os.replace(faded, wav_path)
    m2 = measure_loudness(wav_path)
    print(f"  校正后实测: {m2.get('input_i')} LUFS / TP {m2.get('input_tp')} dBTP")
    return m2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, choices=["A", "B"])
    ap.add_argument("--segments", type=int, default=3, help="每镜段数")
    args = ap.parse_args()

    sm = SAMPLES[args.sample]
    shot_names = sm["shots"]
    by_name = {s["name"]: s for s in SHOTS}
    os.makedirs(OUT_DIR, exist_ok=True)
    wd = f"{WORK_DIR}_{args.sample}"
    if os.path.exists(wd):
        shutil.rmtree(wd)
    os.makedirs(wd)

    # ---------- 1. 镜级拼接 ----------
    print(f"=== 样本 {args.sample}：{sm['name']} ===", flush=True)
    print("=== 1. 镜级拼接 ===", flush=True)
    shots = []
    for nm in shot_names:
        s = by_name[nm]
        vp = concat_segments(nm, args.segments)
        if not vp:
            print(f"  [skip] 缺视频 {nm}")
            continue
        # 该镜实际段数 → 帧数 → 时长（按帧算，绝不用秒数）
        nseg = len([k for k in range(args.segments)
                    if os.path.exists(f"{SEGS_DIR}/{nm}_s{k}.mp4")])
        total_frames = nseg * SEG_FRAMES
        seg_dur = total_frames / FPS
        voice_path = speaker = text = None
        for sp, tx in s["dialogue"]:
            cand = f"{VOICE_DIR}/{nm}_{sp}.wav"
            if os.path.exists(cand):
                voice_path, speaker, text = cand, sp, tx
                break
        shots.append({"id": s["id"], "name": nm, "seg_dur": seg_dur,
                      "frames": total_frames, "voice_path": voice_path,
                      "speaker": speaker, "text": text})
        print(f"  [{nm}] {nseg} 段 = {total_frames} 帧 = {seg_dur:.3f}s", flush=True)
    if not shots:
        raise SystemExit("没有可用镜头")

    # ---------- 2. 视频轨 ----------
    print("\n=== 2. 视频轨（无声）===", flush=True)
    seg_paths = []
    for s in shots:
        seg = f"{wd}/v_{s['name']}.mp4"
        vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
              f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
              f"trim=end_frame={s['frames']},setpts=PTS-STARTPTS")
        run([FFMPEG, "-y", "-i", f"{MOTION_DIR}/{s['name']}.mp4", "-vf", vf, "-an",
             "-c:v", "libx264", "-preset", "medium", "-crf", "19",
             "-pix_fmt", "yuv420p", "-r", str(FPS), seg])
        seg_paths.append(seg)
        print(f"  [{s['name']}] {s['frames']} 帧")

    intro = f"{wd}/intro.mp4"
    render_intro_card(intro, sm["title"], sm["subtitle"], INTRO_FRAMES / FPS)
    all_segs = [intro] + seg_paths
    cmd = [FFMPEG, "-y"]
    for p in all_segs:
        cmd += ["-i", p]
    fc = []
    for i in range(len(all_segs)):
        fc.append(f"[{i}:v]fps={FPS},format=yuv420p,setpts=PTS-STARTPTS[v{i}]")
    chain = "".join(f"[v{i}]" for i in range(len(all_segs)))
    fc.append(f"{chain}concat=n={len(all_segs)}:v=1:a=0[vout]")
    video_only = f"{wd}/video_only.mp4"
    cmd += ["-filter_complex", ";".join(fc), "-map", "[vout]",
            "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
            "-pix_fmt", "yuv420p", "-r", str(FPS), video_only]
    run(cmd)
    print(f"  视频轨: {video_only}")

    # ---------- 3. 音频混音（样本级） ----------
    print("\n=== 3. 音频混音（Python 样本级）===", flush=True)
    raw_mix = f"{wd}/mix_raw.wav"
    norm_mix = f"{wd}/mix_norm.wav"
    timeline_path = f"{OUT_DIR}/timeline_{args.sample}.json"
    os.environ["PAD_HEAD"] = str(PAD_HEAD)
    rows = mix_audio.build_full_mix(
        shots, MOTION_DIR, VOICE_DIR, INTRO_FRAMES / FPS, raw_mix, timeline_path)

    print("\n=== 4. 整片响度归一（两遍 loudnorm + 精确校正）===", flush=True)
    ln = two_pass_loudnorm(raw_mix, norm_mix)
    total_frames_pre = INTRO_FRAMES + sum(s["frames"] for s in shots)
    total_dur_pre = total_frames_pre / FPS
    ln = normalize_to_target(norm_mix, total_dur_pre)

    # ---------- 5. 字幕 ----------
    total_frames = INTRO_FRAMES + sum(s["frames"] for s in shots)
    total_dur = total_frames / FPS
    ass_path = f"{wd}/subs.ass"
    ass_txt, n_sub = build_ass(rows, total_dur)
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_txt)
    print(f"\n=== 5. 字幕 {n_sub} 条，总时长 {total_dur:.3f}s / {total_frames} 帧 ===")

    # ---------- 6. 最终封装 ----------
    print("=== 6. 最终输出 ===", flush=True)
    out_name = f"零点四十七单_{sm['name']}.mp4"
    out_path = f"{OUT_DIR}/{out_name}"
    # fade 已烘进 norm_mix（归一化阶段），这里不再叠 alimiter ——
    # loudnorm 已控 TP，再串限幅器会削峰值把响度拉到 -15 LUFS。
    cmd = [FFMPEG, "-y", "-i", video_only, "-i", norm_mix,
           "-map", "0:v", "-map", "1:a",
           "-vf",
           "eq=gamma=1.06:brightness=0.010:contrast=1.04:saturation=1.05,"
           "curves=all='0/0.015 0.25/0.27 1/1',"
           f"ass='{ass_path}'",
           "-c:v", "libx264", "-preset", "slow", "-crf", "19",
           "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-c:a", "aac", "-b:a", "192k", "-ar", str(SR),
           "-frames:v", str(total_frames),
           "-movflags", "+faststart", out_path]
    run(cmd)

    print(f"\n成片: {out_path}")
    print(f"总时长 {total_dur:.3f}s（{total_frames} 帧 @ {FPS}fps）")
    if ln:
        print(f"响度: 输入 {ln.get('input_i')} LUFS -> 目标 {TARGET_LUFS} LUFS "
              f"(TP {ln.get('input_tp')} dBTP)")
    with open(timeline_path, "w", encoding="utf-8") as f:
        json.dump({"sample": args.sample, "name": sm["name"],
                   "total_dur": round(total_dur, 3), "total_frames": total_frames,
                   "fps": FPS, "intro_dur": INTRO_FRAMES / FPS,
                   "target_lufs": TARGET_LUFS, "out": out_path,
                   "video_only": video_only,
                   "shots": rows}, f, ensure_ascii=False, indent=2)
    print(f"时间轴: {timeline_path}")


if __name__ == "__main__":
    main()
