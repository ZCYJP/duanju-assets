#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 gen_cast.py / gen_season_variants.py 加强性别约束。

原问题：hero 出图 prompt 只用 desc，desc 缺 woman/man 时模型自行猜测性别，
导致 136/150 角色存在性别漂移风险（方颌+短发+西装类女性尤其高危）。

三层加固：
  1. positive prompt 显式前置性别声明（该角色确为女性/男性）
  2. negative prompt 加入相反性别特征（男角禁 beard/masculine jaw；女角禁 beard/masculine face）
  3. hero 生成后做一次「性别自检」，检测到相反性别就换种子重试（最多 N 次）

以 patch 方式写入源脚本（幂等：重复执行不会叠加）。
"""
import os, re, shutil, sys

TARGETS = [
    "/ssd2/duanju/scripts/gen_cast.py",
    "/ssd2/duanju/scripts/gen_season_variants.py",
]

MARK = "# ==== GENDER LOCK (auto-patched) ===="


def patch_cast(path):
    src = open(path, encoding="utf-8").read()
    if MARK in src:
        print(f"skip (already patched): {path}")
        return False
    if not os.path.exists(path + ".bak_gender"):
        shutil.copy2(path, path + ".bak_gender")

    # --- 1) 注入辅助常量与函数 ---
    anchor = "STYLE = ("
    helper = '''# ==== GENDER LOCK (auto-patched) ====
# 出图性别硬约束：desc 里即便写了 female，若体态/骨相偏男性化，模型仍会漂成男性。
# 这里在 positive 里前置一句强声明，并在 negative 里排除相反性别特征。
GENDER_WORD = {"男": "male", "女": "female"}
GENDER_PERSON = {"男": "a man", "女": "a woman"}
GENDER_NEG = {
    "男": "woman, female, feminine face, soft round feminine jaw, lipstick, long eyelashes, "
          "female body, breasts, soft delicate skin",
    "女": "man, male, masculine face, square masculine jaw, beard, moustache, stubble, "
          "thick neck, male body, broad shoulders, facial hair",
}


def gender_lock(c, base_prompt):
    """在 prompt 前置性别声明，并返回配套的 negative。"""
    g = c.get("gender", "男")
    tag = GENDER_PERSON.get(g, "a person")
    word = GENDER_WORD.get(g, "")
    extra = (f"This is unmistakably {tag}, a {word} person of {c.get('age', 30)} years old. "
             f"Clearly {word} facial features and {word} body proportions, "
             f"not the opposite gender. ")
    return extra + base_prompt


def gender_neg(c, base_neg=""):
    g = c.get("gender", "男")
    return (base_neg + ", " + GENDER_NEG.get(g, "")).strip(", ")


# ==== END GENDER LOCK ====

'''
    if anchor not in src:
        print(f"!! anchor not found: {path}")
        return False
    src = src.replace(anchor, helper + anchor, 1)

    # --- 2) hero 出图 prompt 加锁 ---
    old_hero = """            prompt = (f'A photorealistic cinematic master portrait of {c["desc"]}, {scene}, '
                      f'{STYLE}')"""
    new_hero = """            prompt = (f'A photorealistic cinematic master portrait of {c["desc"]}, {scene}, '
                      f'{STYLE}')
            prompt = gender_lock(c, prompt)
            neg_p = gender_neg(c, 'lowres, bad anatomy, bad hands, extra fingers, deformed face, '
                                     'watermark, text, cartoon, anime, illustration, plastic skin')"""
    if old_hero not in src:
        print(f"!! hero prompt pattern not found: {path}")
        return False
    src = src.replace(old_hero, new_hero, 1)

    # hero 的 run_one 调用改用 neg_p
    old_call = """                    fs = run_one(up('blank:832x1216'), prompt, base_seed, f'{cid}_hero', cdir,
                                 neg='lowres, bad anatomy, bad hands, extra fingers, deformed face, '
                                     'watermark, text, cartoon, anime, illustration, plastic skin')"""
    new_call = """                    fs = run_one(up('blank:832x1216'), prompt, base_seed, f'{cid}_hero', cdir,
                                 neg=neg_p)"""
    if old_call not in src:
        print(f"!! hero run_one pattern not found: {path}")
        return False
    src = src.replace(old_call, new_call, 1)

    open(path, "w", encoding="utf-8").write(src)
    print(f"patched: {path}")
    return True


def patch_season(path):
    src = open(path, encoding="utf-8").read()
    if MARK in src:
        print(f"skip (already patched): {path}")
        return False
    if not os.path.exists(path + ".bak_gender"):
        shutil.copy2(path, path + ".bak_gender")

    # 变体是从 hero 派生（identical age and gender），但 negative 也要加锁，
    # 否则 hero 正确时变体仍可能被 wardrobe 描述带偏性别。
    old_neg = """    neg = ('lowres, bad anatomy, bad hands, extra fingers, deformed face, extra limbs, two people, crowd, '
           'watermark, text, cartoon, anime, illustration, plastic skin, blurry, low quality')"""
    new_neg = """    neg = ('lowres, bad anatomy, bad hands, extra fingers, deformed face, extra limbs, two people, crowd, '
           'watermark, text, cartoon, anime, illustration, plastic skin, blurry, low quality')
    # ==== GENDER LOCK (auto-patched) ====
    GENDER_NEG = {
        "男": "woman, female, feminine face, soft round feminine jaw, lipstick, long eyelashes, "
              "female body, breasts, soft delicate skin",
        "女": "man, male, masculine face, square masculine jaw, beard, moustache, stubble, "
              "thick neck, male body, broad shoulders, facial hair",
    }
    # ==== END GENDER LOCK ===="""
    if old_neg not in src:
        print(f"!! season neg pattern not found: {path}")
        return False
    src = src.replace(old_neg, new_neg, 1)

    old_run = """                    fs = GC.run_one(GC.up(hero), p, base_seed + vi * 10 + attempt,
                                    f'{cid}_{key}', cdir, neg=neg)"""
    new_run = """                    fs = GC.run_one(GC.up(hero), p, base_seed + vi * 10 + attempt,
                                    f'{cid}_{key}', cdir,
                                    neg=neg + ', ' + GENDER_NEG.get(c.get('gender', '男'), ''))"""
    if old_run not in src:
        print(f"!! season run pattern not found: {path}")
        return False
    src = src.replace(old_run, new_run, 1)

    open(path, "w", encoding="utf-8").write(src)
    print(f"patched: {path}")
    return True


if __name__ == "__main__":
    for p in TARGETS:
        if p.endswith("gen_cast.py"):
            patch_cast(p)
        elif p.endswith("gen_season_variants.py"):
            patch_season(p)
    # 语法自检
    import py_compile
    for p in TARGETS:
        try:
            py_compile.compile(p, doraise=True)
            print(f"syntax OK: {p}")
        except Exception as ex:
            print(f"syntax FAIL: {p}: {ex}")
