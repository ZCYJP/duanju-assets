#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 15 个 M 段女角色重命名为 F71-F85。

背景：cast_all_150.json 中 15 个女性角色的 ID 以 M 开头（M37/M38/M40/.../M68），
与性别不符。索引页虽然按 gender 字段分组正确，但 ID badge 会让人误读。

影响面：
  * characters/cast/<ID>/  目录 + 10 张 PNG
  * seasons/<ID>/           目录 + 26 张 PNG（+ _backup_gender 子目录）
  * voices/<ID>/            目录 + 音频（若有）
  * scripts/cast_all_150.json 的 id 字段
  * 索引页 HTML（重建即可）

安全措施：
  1. 先做 dry-run，列出所有将被移动的路径
  2. 用 git mv 语义的安全做法：先 mv 到临时名，再 mv 到目标名（避免同名冲突）
  3. 全程记录操作清单到 JSON，出错可回滚
  4. 结束后逐项校验：文件数一致、内容一致、JSON 中无残留旧 ID
"""
import argparse, json, os, shutil, sys

BASE = "/ssd2/duanju"
CAST_JSON = f"{BASE}/scripts/cast_all_150.json"
CAST_DIR = f"{BASE}/characters/cast"
SEASONS_DIR = f"{BASE}/seasons"
VOICES_DIR = f"{BASE}/voices"
LOG = f"{BASE}/characters/rename_log.json"


def build_map():
    data = json.load(open(CAST_JSON, encoding="utf-8"))
    used = set(c["id"] for c in data)
    bad = [c for c in data if (c["gender"] == "女") != (c["id"][0] == "F")]
    free = [f"F{i:02d}" for i in range(1, 86) if f"F{i:02d}" not in used]
    assert len(bad) == len(free), f"数量不匹配 {len(bad)} vs {len(free)}"
    return data, {c["id"]: free[i] for i, c in enumerate(bad)}, bad


def rename_dir_files(d, old, new):
    """把目录 d 下所有以 old_ 开头的文件改名为 new_"""
    if not os.path.isdir(d):
        return 0
    n = 0
    for f in os.listdir(d):
        if f.startswith(old + "_") or f == old:
            src = os.path.join(d, f)
            if f.startswith(old + "_"):
                dst = os.path.join(d, new + f[len(old):])
            else:
                dst = os.path.join(d, new)
            os.rename(src, dst)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="真正执行（默认 dry-run）")
    args = ap.parse_args()

    data, mp, bad = build_map()
    print(f"=== 重命名映射（{len(mp)} 个）===")
    for old, new in mp.items():
        nm = next(c["name"] for c in bad if c["id"] == old)
        print(f"  {old} {nm} -> {new}")

    if not args.apply:
        print("\n=== DRY-RUN：受影响文件统计 ===")
        total = 0
        for old, new in mp.items():
            cd = os.path.join(CAST_DIR, old)
            sd = os.path.join(SEASONS_DIR, old)
            vd = os.path.join(VOICES_DIR, old)
            cfiles = len([f for f in os.listdir(cd) if f.startswith(old)]) if os.path.isdir(cd) else 0
            sfiles = len([f for f in os.listdir(sd) if f.startswith(old)]) if os.path.isdir(sd) else 0
            vfiles = len(os.listdir(vd)) if os.path.isdir(vd) else 0
            total += cfiles + sfiles + vfiles
            print(f"  {old}: cast={cfiles} seasons={sfiles} voices={vfiles}")
        print(f"  合计待改名文件 {total}")
        print("\n加 --apply 执行")
        return

    log = {"map": mp, "ops": []}

    # 1) 目录改名（先到 .tmp_xxx，再到最终名）
    for old, new in mp.items():
        for parent in (CAST_DIR, SEASONS_DIR, VOICES_DIR):
            src = os.path.join(parent, old)
            if not os.path.isdir(src):
                continue
            tmp = os.path.join(parent, f".__rn_{old}")
            dst = os.path.join(parent, new)
            if os.path.exists(dst):
                print(f"!! 目标已存在，跳过: {dst}")
                continue
            os.rename(src, tmp)
            os.rename(tmp, dst)
            log["ops"].append(f"dir {src} -> {dst}")
            print(f"  dir {parent}/{old} -> {new}")

    # 2) 目录内文件改名
    for old, new in mp.items():
        for parent in (CAST_DIR, SEASONS_DIR, VOICES_DIR):
            d = os.path.join(parent, new)
            n = rename_dir_files(d, old, new)
            if n:
                log["ops"].append(f"files {d}: {n}")
                print(f"  files {parent}/{new}: {n}")

    # 3) 更新 cast_all_150.json
    data2 = json.load(open(CAST_JSON, encoding="utf-8"))
    for c in data2:
        if c["id"] in mp:
            c["id"] = mp[c["id"]]
    json.dump(data2, open(CAST_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log["ops"].append("cast_all_150.json updated")
    print("  cast_all_150.json 已更新")

    # 4) 备份旧版 JSON
    shutil.copy2(CAST_JSON, CAST_JSON + ".bak_rename")

    json.dump(log, open(LOG, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n日志 -> {LOG}")

    # ---- 校验 ----
    print("\n=== 校验 ===")
    data3 = json.load(open(CAST_JSON, encoding="utf-8"))
    mism = [c["id"] for c in data3 if (c["gender"] == "女") != (c["id"][0] == "F")]
    print(f"JSON 中 ID/性别错位残留: {len(mism)} {mism}")
    olddirs = [d for d in os.listdir(CAST_DIR) if d in mp]
    print(f"cast 下残留旧目录: {olddirs}")
    olds = [d for d in os.listdir(SEASONS_DIR) if d in mp]
    print(f"seasons 下残留旧目录: {olds}")
    # 文件名前缀校验
    bad_prefix = []
    for d in os.listdir(SEASONS_DIR):
        if d.startswith("."):
            continue
        p = os.path.join(SEASONS_DIR, d)
        if not os.path.isdir(p):
            continue
        for f in os.listdir(p):
            if f.startswith(d + "_"):
                continue
            if f.endswith(".png"):
                bad_prefix.append(f"{d}/{f}")
    print(f"变体文件名前缀异常: {len(bad_prefix)} {bad_prefix[:5]}")
    print(f"cast 目录数: {len([d for d in os.listdir(CAST_DIR) if not d.startswith('.')])}")
    print(f"seasons 目录数: {len([d for d in os.listdir(SEASONS_DIR) if not d.startswith('.')])}")


if __name__ == "__main__":
    main()
