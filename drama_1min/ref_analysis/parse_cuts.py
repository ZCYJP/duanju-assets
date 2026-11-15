import re, statistics, json, math

p = "/tmp/refan/scenes/cuts_t020.txt"
txt = open(p, encoding="utf-8", errors="ignore").read().splitlines()
cuts = []
i = 0
while i < len(txt):
    m = re.search(r"pts_time:([0-9.]+)", txt[i])
    if m:
        t = float(m.group(1))
        sc = 0.0
        if i + 1 < len(txt) and "scene_score" in txt[i + 1]:
            mm = re.search(r"scene_score=([0-9.]+)", txt[i + 1])
            if mm:
                sc = float(mm.group(1))
        cuts.append((t, sc))
    i += 1

DUR = 1639.621
bounds = [0.0] + [t for t, _ in cuts] + [DUR]
lens = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
lens_pos = [x for x in lens if x > 0.05]


def pct(a, q):
    a = sorted(a)
    k = (len(a) - 1) * q
    f = math.floor(k)
    c = math.ceil(k)
    return a[f] if f == c else a[f] + (a[c] - a[f]) * (k - f)


print("总时长 %.1fs  镜头数 %d  平均镜长 %.3fs" % (DUR, len(lens_pos), statistics.mean(lens_pos)))
print("中位 %.3fs  P10 %.3fs  P25 %.3fs  P75 %.3fs  P90 %.3fs  P95 %.3fs  max %.3fs  min %.3fs" % (
    pct(lens_pos, 0.5), pct(lens_pos, 0.1), pct(lens_pos, 0.25),
    pct(lens_pos, 0.75), pct(lens_pos, 0.9), pct(lens_pos, 0.95),
    max(lens_pos), min(lens_pos)))

hist = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 2.5), (2.5, 3),
        (3, 4), (4, 5), (5, 6), (6, 8), (8, 12), (12, 999)]
print("\n镜头长度分布：")
for lo, hi in hist:
    c = sum(1 for x in lens_pos if lo <= x < hi)
    bar = "#" * int(c / len(lens_pos) * 160)
    print("  [%5.1f,%6.1f)s : %5d 镜 %5.1f%%  %s" % (lo, hi, c, c / len(lens_pos) * 100, bar))

n = len(lens_pos)
print("\n<=1s  %.1f%%   <=2s %.1f%%   <=3s %.1f%%   >=4s %.1f%%   >=5s %.1f%%" % (
    sum(1 for x in lens_pos if x <= 1) / n * 100,
    sum(1 for x in lens_pos if x <= 2) / n * 100,
    sum(1 for x in lens_pos if x <= 3) / n * 100,
    sum(1 for x in lens_pos if x >= 4) / n * 100,
    sum(1 for x in lens_pos if x >= 5) / n * 100))
print("切换密度 = %.3f cuts/s" % (n / DUR))

core = [x for x in lens_pos if 1.0 <= x <= 20]
print("核心区(1~20s): 平均 %.3fs 中位 %.3fs n=%d" % (
    statistics.mean(core), pct(core, 0.5), len(core)))

# 我们的成片对比数据
ours = [4.9167] * 11 + [5.1667]
print("\n[我们的成片] 12 镜 平均 %.3fs  <=3s 占比 0.0%%" % statistics.mean(ours))

json.dump({"cuts": [[t, s] for t, s in cuts], "lens": lens_pos, "dur": DUR},
          open("/tmp/refan/cuts.json", "w"))
print("\nwrote /tmp/refan/cuts.json")
