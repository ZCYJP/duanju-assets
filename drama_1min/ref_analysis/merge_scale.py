"""汇总景别分布 + 与我们成片对照"""
import json
import glob
import numpy as np
from collections import Counter

rows = []
for fp in sorted(glob.glob("/tmp/refan/shards/scale_*.json")):
    rows += json.load(open(fp))
print("总帧", len(rows))

# 帧号 → 时间（2fps）
for r in rows:
    n = int(r["f"].split("_")[1].split(".")[0])
    r["t"] = (n - 1) / 2.0

N = len(rows)
cc = Counter(r["cls"] for r in rows)
print("\n=== 参考片景别分布（SCRFD，人脸高占画面比）===")
for k in ["大特写/特写", "特写", "近景", "中景", "中全景", "全景/远景", "无脸"]:
    v = cc.get(k, 0)
    print("  %-12s : %5d  %5.1f%%  %s" % (k, v, v / N * 100, "#" * int(v / N * 140)))

fp_rows = [r for r in rows if r["n"] > 0]
print("\n有人脸帧 %d (%.1f%%)  无人脸帧 %d (%.1f%%)" % (
    len(fp_rows), len(fp_rows) / N * 100, N - len(fp_rows), (N - len(fp_rows)) / N * 100))

fh = np.array([r["fh"] for r in fp_rows])
print("\n人脸高占比 均值 %.3f 中位 %.3f P10 %.3f P25 %.3f P75 %.3f P90 %.3f" % (
    fh.mean(), np.median(fh), *np.percentile(fh, [10, 25, 75, 90])))
print("  >0.11(近景及以上) 占有人脸帧 %.1f%%" % ((fh > 0.11).mean() * 100))
print("  >0.06(中景及以上) 占有人脸帧 %.1f%%" % ((fh > 0.06).mean() * 100))
print("  <0.06(中全景及以下) 占有人脸帧 %.1f%%" % ((fh < 0.06).mean() * 100))

nf = Counter(r["n"] for r in rows)
print("\n=== 同框人数 ===")
for k in sorted(nf):
    print("  %d 人 : %5d 帧 %5.1f%%" % (k, nf[k], nf[k] / N * 100))

cx = np.array([r["cx"] for r in fp_rows])
cy = np.array([r["cy"] for r in fp_rows])
print("\n人物水平位置: 左1/3 %.1f%%  中1/3 %.1f%%  右1/3 %.1f%%" % (
    (cx < 1/3).mean()*100, ((cx >= 1/3) & (cx < 2/3)).mean()*100, (cx >= 2/3).mean()*100))
print("人物垂直位置: 上1/3 %.1f%%  中1/3 %.1f%%  下1/3 %.1f%%" % (
    (cy < 1/3).mean()*100, ((cy >= 1/3) & (cy < 2/3)).mean()*100, (cy >= 2/3).mean()*100))

br = np.array([r["bright"] for r in rows])
print("\n亮度 均值 %.1f P5 %.1f P95 %.1f" % (br.mean(), np.percentile(br, 5), np.percentile(br, 95)))

# 按镜头切点看景别切换（相邻镜头景别是否跳变）
cuts = json.load(open("/tmp/refan/cuts.json"))
ct = [c[0] for c in cuts["cuts"]]
print("\n=== 景别切换频率（以 2fps 帧近似镜头）===")
# 找出每个镜头中点的帧
mid_pts = []
bounds = [0.0] + ct + [1639.621]
for i in range(len(bounds)-1):
    mid_pts.append((bounds[i]+bounds[i+1])/2)
cls_by_t = []
for r in rows:
    cls_by_t.append((r["t"], r["cls"]))
def lookup(t):
    best = None
    for tt, c in cls_by_t:
        if abs(tt - t) < 0.3:
            best = c
    return best
seq = []
for t in mid_pts:
    c = lookup(t)
    if c:
        seq.append(c)
if len(seq) > 2:
    changes = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i-1])
    print("  镜头序列长度 %d，景别发生切换 %d 次 (%.1f%%)" % (
        len(seq), changes, changes / len(seq) * 100))
    from collections import Counter as C2
    print("  最常见景别对（前一镜→后一镜）:")
    pairs = C2("%s→%s" % (seq[i-1], seq[i]) for i in range(1, len(seq)))
    for k, v in pairs.most_common(10):
        print("    %-28s %4d  %.1f%%" % (k, v, v/len(seq)*100))

json.dump({"dist": dict(cc), "n": N}, open("/tmp/refan/scale_summary.json", "w"), ensure_ascii=False, indent=2)
