#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复 cast_all_150.json 的 desc 字段性别词缺失。

根因：出图 prompt 是 `A photorealistic cinematic master portrait of {desc}, {scene}, {STYLE}`，
desc 里没有 woman/man 时，模型只能靠职业与体态特征猜性别 → 方颌短发+西装类女性角色
被渲染成男性。136/150 角色受影响。

规则：
  1. desc 开头形如 `a {age}-year-old Chinese {occupation}, {features...}`
     - 有职业词：在职业前插入 female/male
       `a 34-year-old Chinese manufacturing entrepreneur` → `a 34-year-old Chinese female manufacturing entrepreneur`
     - 无职业词（紧跟逗号）：插入 woman/man
       `a 30-year-old Chinese, tall and lean` → `a 30-year-old Chinese man, tall and lean`
  2. 已有 woman/female/man/male 的不动
  3. gender 为女→female/woman，为男→male/man
写入前备份原文件到 cast_all_150.json.bak_gender
"""
import json, os, re, shutil, sys

SRC = "/ssd2/duanju/scripts/cast_all_150.json"
BAK = SRC + ".bak_gender"

FEM_WORDS = re.compile(r"\b(woman|female|lady|girl|women|her|she)\b", re.I)
MAL_WORDS = re.compile(r"\b(man|male|gentleman|boy|men|his|he)\b", re.I)

# 匹配 `a {age}-year-old Chinese` 之后到第一个逗号
HEAD = re.compile(r"^(a\s+\d+-year-old\s+Chinese)([^,]*)(,.*)$", re.S)


def inject(desc, gender):
    if gender == "女":
        if FEM_WORDS.search(desc):
            return desc, "skip-already-female"
        word, noun = "female", "woman"
    else:
        if MAL_WORDS.search(desc):
            return desc, "skip-already-male"
        word, noun = "male", "man"

    m = HEAD.match(desc)
    if not m:
        return desc, "skip-no-match"
    head, occ, rest = m.group(1), m.group(2).strip(), m.group(3)
    if occ:
        # 有职业词：职业前加 female/male
        new = f"{head} {word} {occ}{rest}"
        return new, "insert-adjective"
    # 无职业词：逗号前加 woman/man
    new = f"{head} {noun}{rest}"
    return new, "insert-noun"


def main():
    data = json.load(open(SRC, encoding="utf-8"))
    if not os.path.exists(BAK):
        shutil.copy2(SRC, BAK)
        print(f"已备份 -> {BAK}")
    else:
        print(f"备份已存在，沿用: {BAK}")

    stats = {}
    changed = []
    for c in data:
        new, how = inject(c["desc"], c["gender"])
        stats[how] = stats.get(how, 0) + 1
        if new != c["desc"]:
            changed.append((c["id"], c["name"], c["gender"], c["desc"], new))
            c["desc"] = new

    json.dump(data, open(SRC, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("统计:", stats)
    print(f"修改 {len(changed)} 个角色\n")
    for cid, name, g, old, new in changed[:8]:
        print(f"[{cid}] {name} ({g})")
        print(f"  旧: {old[:110]}")
        print(f"  新: {new[:110]}")
    if len(changed) > 8:
        print(f"  ... 其余 {len(changed)-8} 个已修改")

    # 复检
    data2 = json.load(open(SRC, encoding="utf-8"))
    bad = []
    for c in data2:
        if c["gender"] == "女" and not FEM_WORDS.search(c["desc"]):
            bad.append((c["id"], "女", c["desc"][:70]))
        if c["gender"] == "男" and not MAL_WORDS.search(c["desc"]):
            bad.append((c["id"], "男", c["desc"][:70]))
    print(f"\n复检：150 角色中仍有 {len(bad)} 个缺性别词")
    for b in bad:
        print("  ", b)


if __name__ == "__main__":
    main()
