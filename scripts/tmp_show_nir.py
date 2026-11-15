import json
import sys

d = json.load(open(sys.argv[1]))
ch = [r for r in d if r.get("changed")]
nr = [r for r in ch if not r.get("in_range")]
print("changed:", len(ch), " not_in_range:", len(nr))
for r in sorted(nr, key=lambda x: -x.get("ratio_before", 0)):
    print("  {} {} r {:.2f} -> {:.2f} rounds={} target={:.3f}".format(
        r["cid"], r["emotion"], r["ratio_before"], r["ratio_after"],
        r["rounds"], r.get("factor_total", 0)))
