#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补跑季节变体质检：只处理 qc_shard_*.json 里没覆盖到的变体。

背景：4 分片质检是在变体还没全部生成完时启动的，只覆盖了 3479/3900。
本脚本扫描全部 seasons/*.png，跳过已质检过的，跑完后合并成完整报告。

用法：python3 tmp_vl_qc_season_gap.py --of 4 --shard 0
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, "/ssd2/duanju/scripts")
import tmp_vl_qc_season as Q

SEASON_ROOT = "/ssd2/duanju/seasons"
CAST_ROOT = "/ssd2/duanju/characters/cast"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--of", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out", default="/ssd2/duanju/seasons/qc_gap.json")
    args = ap.parse_args()

    # 已质检过的键
    done = set()
    for f in sorted(glob.glob(f"{SEASON_ROOT}/qc_shard_*.json")):
        if ".arbitrated" in f:
            continue
        for r in json.load(open(f, encoding="utf-8")):
            done.add((r["cid"], r["variant"]))

    todo = []
    for p in sorted(glob.glob(f"{SEASON_ROOT}/*/*.png")):
        cid = os.path.basename(os.path.dirname(p))
        name = os.path.basename(p)[:-4]
        if (cid, name) in done:
            continue
        hero = f"{CAST_ROOT}/{cid}/{cid}_hero.png"
        if not os.path.exists(hero):
            continue
        todo.append((cid, name, p, hero))

    todo = [x for i, x in enumerate(todo) if i % args.of == args.shard]
    print(f"[data] 已质检 {len(done)}, 待补 {len(todo)} (shard {args.shard}/{args.of})",
          flush=True)

    report = []
    import time
    t0 = time.time()
    for n, (cid, name, vp, hero) in enumerate(todo):
        try:
            v, raw = Q.cross(hero, vp)
        except Exception as ex:
            v, raw = "ERROR", [str(ex)[:120]]
        report.append({"cid": cid, "variant": name, "verdict": v, "raw": raw})
        if (n + 1) % 20 == 0:
            el = time.time() - t0
            print(f"  {n+1}/{len(todo)} ({el:.0f}s)", flush=True)
            json.dump(report, open(args.out, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)

    json.dump(report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    same = sum(1 for r in report if r["verdict"] == "是")
    diff = sum(1 for r in report if r["verdict"] == "否")
    sus = sum(1 for r in report if r["verdict"] == "存疑")
    print(f"[done] 同人={same} 换人={diff} 存疑={sus} 总计={len(report)} -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
