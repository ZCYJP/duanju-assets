"""参考片台词节奏分析：ASR 拿台词文本+时间戳 → 统计语速/句长/信息密度/情绪词"""
import json
import re
import numpy as np
import subprocess
import os
from faster_whisper import WhisperModel

SR = 16000
DUR = 1639.621
model = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=16)
print("model loaded")

# 分块 ASR：每块 120s，重叠 2s
CHUNK = 120.0
OVER = 2.0
segs = []
t = 0.0
while t < DUR:
    cmd = ["/usr/local/bin/ffmpeg", "-v", "error", "-ss", "%.3f" % t, "-i", "/tmp/ref.mp4",
           "-t", "%.3f" % (CHUNK + OVER), "-vn", "-ac", "1", "-ar", str(SR),
           "-f", "wav", "-y", "/tmp/refan/_chunk.wav"]
    subprocess.run(cmd, check=True)
    ss, info = model.transcribe("/tmp/refan/_chunk.wav", language="zh", beam_size=5,
                                vad_filter=True, condition_on_previous_text=False)
    for s in ss:
        if s.text.strip():
            segs.append({"start": round(t + s.start, 3), "end": round(t + s.end, 3),
                         "text": s.text.strip()})
    print("  chunk @%.0fs -> %d segs" % (t, len(segs)), flush=True)
    t += CHUNK

json.dump(segs, open("/tmp/refan/asr_segs.json", "w"), ensure_ascii=False, indent=1)
print("total segs", len(segs))

texts = [s["text"] for s in segs]
allt = "".join(texts)
cn = len(re.findall(r"[\u4e00-\u9fff]", allt))
print("\n=== 台词密度 ===")
print("总中文字数 %d / 时长 %.1fs = %.2f 字/秒" % (cn, DUR, cn / DUR))
print("片段数 %d  平均每段 %.2fs  平均每段 %.1f 字" % (
    len(segs), np.mean([s["end"] - s["start"] for s in segs]), cn / max(len(segs), 1)))

lens = [s["end"] - s["start"] for s in segs]
print("段长 中位 %.2fs  P25 %.2fs  P75 %.2fs  P90 %.2fs  max %.2fs" % (
    np.median(lens), *np.percentile(lens, [25, 75, 90]), max(lens)))
shorth = sum(1 for x in lens if x <= 5)
print("<=5s 台词段占比 %.1f%%" % (shorth / len(lens) * 100))

# 语速：每秒中文字数（逐段）
sp = [cn_seg / max(l, 0.1) for cn_seg, l in
      zip([len(re.findall(r"[\u4e00-\u9fff]", s["text"])) for s in segs], lens)]
print("语速 中位 %.2f 字/秒  P10 %.2f  P90 %.2f" % (
    np.median(sp), np.percentile(sp, 10), np.percentile(sp, 90)))

# 句末标点分布 → 冲突/情绪
print("\n=== 情绪/冲突词频（每万字） ===")
KW = {
    "疑问": "吗|什么|怎么|为什么|凭什么|难道|是不是|谁|哪儿",
    "愤怒": "滚|闭嘴|你敢|混蛋|别想|休想|凭什么|无法容忍|够了",
    "反转": "原来|竟然|居然|没想到|其实|真相|没想到|才发现",
    "命令": "必须|马上|立刻|现在|立刻|给我|听好了|记住",
    "金钱": "钱|块|万|元|转账|账|付|欠|还",
    "关系": "妈|爸|女儿|儿子|老婆|老公|婆婆|媳妇|亲|家",
}
for k, pat in KW.items():
    n = len(re.findall(pat, allt))
    print("  %-4s : %5d 次  (%.1f/万字)" % (k, n, n / max(cn, 1) * 10000))

print("\n=== 开头 40 条台词（看叙事钩子） ===")
for s in segs[:40]:
    print("  %7.2f  %s" % (s["start"], s["text"][:60]))
print("\n=== 台词间隔（无对白空档） ===")
gaps = []
for i in range(1, len(segs)):
    gaps.append(segs[i]["start"] - segs[i - 1]["end"])
if gaps:
    g = np.array(gaps)
    print("  间隔 中位 %.2fs  均值 %.2fs  P90 %.2fs  max %.2fs  >1s占比 %.1f%%" % (
        np.median(g), g.mean(), np.percentile(g, 90), g.max(), (g > 1).mean() * 100))
    print("  平均台词-切镜比：每 %.2fs 一句台词" % (DUR / max(len(segs), 1)))
