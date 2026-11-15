"""分析参考片多人镜头的构成：反打/过肩/并置 —— 用人脸横向分布判断"""
import json
import glob
import numpy as np

rows = []
for fp in sorted(glob.glob("/tmp/refan/shards/scale_*.json")):
    rows += json.load(open(fp))
for r in rows:
    n = int(r["f"].split("_")[1].split(".")[0])
    r["t"] = (n - 1) / 2.0

multi = [r for r in rows if r["n"] >= 2]
print("多人帧 %d / %d (%.1f%%)" % (len(multi), len(rows), len(multi)/len(rows)*100))

# 多人帧中，最大人脸的横向位置分布
print("\n多人帧中主角(最大脸)水平位置：")
cx = np.array([r["cx"] for r in multi])
print("  左1/3 %.1f%%  中1/3 %.1f%%  右1/3 %.1f%%  → 左右分布=过肩/反打特征" % (
    (cx < 1/3).mean()*100, ((cx>=1/3)&(cx<2/3)).mean()*100, (cx>=2/3).mean()*100))

# 主角人脸大小在多人帧
fh = np.array([r["fh"] for r in multi])
print("多人帧主角人脸高占比 中位 %.3f  （>0.18=特写，0.11~0.18=近景）" % np.median(fh))
print("  特写以上占比 %.1f%%  近景 %.1f%%  中景及以下 %.1f%%" % (
    (fh>=0.18).mean()*100, ((fh>=0.11)&(fh<0.18)).mean()*100, (fh<0.11).mean()*100))

# 多人帧的出现间隔（对话场景：镜头来回切）
multis = sorted(r["t"] for r in multi)
if multis:
    gaps = []
    for i in range(1, len(multis)):
        g = multis[i]-multis[i-1]
        if g < 60:
            gaps.append(g)
    g = np.array(gaps)
    print("\n多人帧连续簇：簇数 %d，平均簇时长 %.2fs" % (
        len([1 for i in range(1,len(multis)) if multis[i]-multis[i-1] > 6]),
        np.mean([1]) if len(multis)==0 else 0))
    # 手动算簇
    clusters = []
    st = multis[0]; prev = multis[0]
    for t in multis[1:]:
        if t - prev > 6:
            clusters.append((st, prev))
            st = t
        prev = t
    clusters.append((st, prev))
    cl = [c[1]-c[0] for c in clusters]
    print("  对话簇 %d 个  平均时长 %.2fs  中位 %.2fs  最长 %.2fs" % (
        len(clusters), np.mean(cl), np.median(cl), max(cl)))
    print("  每簇平均含 %.1f 个多人帧（即每簇 %.1f 秒）" % (
        np.mean([sum(1 for t in multis if a <= t <= b) for a, b in clusters]), np.mean(cl)))

# 单人帧的人脸大小
single = [r for r in rows if r["n"] == 1]
sfh = np.array([r["fh"] for r in single])
print("\n单人帧 %d，人脸高中位 %.3f" % (len(single), np.median(sfh)))
print("  单人帧中 特写以上 %.1f%%  近景 %.1f%%  中景以下 %.1f%%" % (
    (sfh>=0.18).mean()*100, ((sfh>=0.11)&(sfh<0.18)).mean()*100, (sfh<0.11).mean()*100))

# 景别跳变：相邻帧（0.5s 粒度）景别变化
seq = sorted([(r["t"], r["cls"]) for r in rows])
ch = sum(1 for i in range(1, len(seq)) if seq[i][1] != seq[i-1][1])
print("\n相邻 0.5s 景别变化率 %.1f%% （越高越有剪辑感）" % (ch/len(seq)*100))
