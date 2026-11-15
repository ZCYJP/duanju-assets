#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为指定角色强制重生成 26 张时令变体（性别修复后同步）。

背景：变体是从 hero 图 img2img 派生的，hero 性别修正后旧变体仍带错误长相。
gen_season_variants.py 会跳过已存在文件，所以这里先备份再强制重生成。

用法: python tmp_regen_seasons_for_ids.py --ids F37 F36 ... --url http://127.0.0.1:8188
"""
import argparse, importlib, json, os, shutil, sys, time

sys.path.insert(0, '/ssd2/short_drama/dh_scripts')
sys.path.insert(0, '/ssd2/duanju/scripts')
import comfy
import gen_cast as GC
from season_scenes import VARIANTS
import gen_season_variants as GSV

CAST = "/ssd2/duanju/scripts/cast_all_150.json"
ROOT = "/ssd2/duanju/seasons"
CAST_ROOT = "/ssd2/duanju/characters/cast"


def instance_alive(url, timeout=8):
    try:
        import urllib.request
        with urllib.request.urlopen(url.rstrip("/") + "/system_stats", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--url", required=True)
    ap.add_argument("--backup", action="store_true", default=True)
    args = ap.parse_args()

    os.environ["COMFY_URL"] = args.url
    importlib.reload(comfy)
    importlib.reload(GC)
    importlib.reload(GC.comfy)
    importlib.reload(GSV)

    cast = {c["id"]: c for c in json.load(open(CAST, encoding="utf-8"))}
    neg_base = ('lowres, bad anatomy, bad hands, extra fingers, deformed face, extra limbs, two people, crowd, '
                'watermark, text, cartoon, anime, illustration, plastic skin, blurry, low quality')

    total_ok = total_fail = 0
    for cid in args.ids:
        c = cast.get(cid)
        if not c:
            print(f"skip unknown {cid}", flush=True)
            continue
        hero = f'{CAST_ROOT}/{cid}/{cid}_hero.png'
        if not os.path.exists(hero):
            print(f"{cid} no hero, skip", flush=True)
            continue
        cdir = os.path.join(ROOT, cid)
        os.makedirs(cdir, exist_ok=True)
        # 备份旧变体
        bdir = os.path.join(cdir, "_backup_gender")
        os.makedirs(bdir, exist_ok=True)
        for f in os.listdir(cdir):
            if f.endswith(".png"):
                dst = os.path.join(bdir, f)
                if not os.path.exists(dst):
                    shutil.copy2(os.path.join(cdir, f), dst)

        sex_off = 0 if cid[0] == 'M' else 50
        base_seed = 1600000 + sex_off + int(cid[1:]) * 100
        scene_base = c.get('scene_en', 'in a softly lit neutral interior with shallow depth of field')
        neg = neg_base + ', ' + GC.GENDER_NEG.get(c.get('gender', '男'), '')

        print(f"\n=== {cid} {c['name']} ({c['gender']}) 26 变体 ===", flush=True)
        for vi, (key, zh, en, mood, clothes) in enumerate(VARIANTS):
            dst = os.path.join(cdir, f'{cid}_{key}.png')
            # 备份后删掉旧图，强制重生成
            if os.path.exists(dst):
                os.remove(dst)
            p = GSV.prompt_for(c, en, mood, clothes, scene_base)
            ok = False
            for attempt in range(4):
                if not instance_alive(args.url):
                    print(f"  {cid} {zh} 实例失联，等待重连...", flush=True)
                    time.sleep(20)
                    continue
                try:
                    fs = GC.run_one(GC.up(hero), p, base_seed + vi * 10 + attempt,
                                    f'{cid}_{key}', cdir, neg=neg)
                    if fs:
                        os.replace(fs[0], dst)
                        ok = True
                        break
                except Exception as ex:
                    print(f"  {cid} {zh} 异常({attempt+1}): {str(ex)[:110]}", flush=True)
                    time.sleep(10)
            if ok:
                total_ok += 1
                print(f"  {cid} {zh}({key}) ✓", flush=True)
            else:
                total_fail += 1
                print(f"  {cid} {zh}({key}) ✗ FAILED", flush=True)

    print(f"\n=== 完成：成功 {total_ok} 失败 {total_fail} ===", flush=True)


if __name__ == "__main__":
    main()
