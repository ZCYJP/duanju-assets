#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 50 人角色库生成拼图总览：all_heroes.jpg（50人主形象）+ 每人10视图sheet。

输出到 <out_root>/sheets/
"""
import json, os, subprocess, sys

DH = "/ssd2/short_drama/dh_scripts"
SHEET = f"{DH}/contact_sheet.py"
PY = "/ssd2/miniconda3/envs/videogen/bin/python"


def sheet(out, cols, tw, items):
    cmd = [PY, SHEET, out, str(cols), str(tw)] + items
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    print("  ", out, os.path.getsize(out) // 1024, "KB")


def main():
    cast_path, cast_root, out_root = sys.argv[1], sys.argv[2], sys.argv[3]
    cast = json.load(open(cast_path, encoding="utf-8"))
    os.makedirs(out_root, exist_ok=True)

    # 1) 全部主形象总览（10列 x 5行）
    items = []
    for c in cast:
        p = f"{cast_root}/{c['id']}/{c['id']}_hero.png"
        if os.path.exists(p):
            items.append(f"{p}={c['id']} {c['name']}")
    print(f"all_heroes: {len(items)} tiles")
    sheet(f"{out_root}/all_heroes.jpg", 10, 300, items)

    # 2) 每人 10 视图 sheet
    VIEWS = ["hero", "headshot", "threequarter", "profile", "back",
             "fullbody", "turnaround", "smile", "angry", "sad"]
    for c in cast:
        cid = c["id"]
        its = []
        for v in VIEWS:
            p = f"{cast_root}/{cid}/{cid}_{v}.png"
            if os.path.exists(p):
                its.append(f"{p}={v}")
        if its:
            sheet(f"{out_root}/{cid}_sheet.jpg", 5, 300, its)

    # 3) tar 打包（便于回传）
    tar = f"{out_root}/cast_all.tar"
    subprocess.run(["tar", "-cf", tar, "-C", cast_root, "."], check=True)
    print("  ", tar, os.path.getsize(tar) // 1024 // 1024, "MB")


if __name__ == "__main__":
    main()
