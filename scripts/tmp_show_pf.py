import json
import sys

d = json.load(open(sys.argv[1]))
ch = [r for r in d if r.get("changed")]
print("changed:", len(ch))
for r in ch[:10]:
    print("{} {} rounds={} ft={:.3f} {:.2f}->{:.2f} in_range={} dur {}->{}".format(
        r["cid"], r["emotion"], r["rounds"], r["factor_total"],
        r["ratio_before"], r["ratio_after"], r["in_range"],
        r["dur_before"], r["dur_after"]))
nr = [r for r in ch if not r["in_range"]]
print("not in range:", len(nr), "/", len(ch))
