#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为所有角色生成 冬/夏 + 24节气 共 26 个季节变体形象。

关键点：
  * 以角色的 master portrait（hero）作为 identity 参考做 image edit，
    保证 26 个变体仍是同一个人（Qwen-Image-Edit 的 image1=主形象）
  * 服装按节气调整（冬棉夏单），场景与光照随节气变化
  * 输出 /ssd2/duanju/seasons/<CID>/<CID>_<variant>.png
  * 支持 --shard/--of 多实例并行
"""
import argparse, importlib, json, os, sys, time

sys.path.insert(0, '/ssd2/short_drama/dh_scripts')
sys.path.insert(0, '/ssd2/duanju/scripts')
import comfy
import gen_cast as GC
from season_scenes import VARIANTS

CAST = "/ssd2/duanju/scripts/cast_all_150.json"
ROOT = "/ssd2/duanju/seasons"


def prompt_for(c, en, mood, clothes, scene_base):
    return (f'Keep the exact same person and the exact same face as in image 1: identical facial features, '
            f'identical hairstyle and hair colour, identical skin tone, identical body build and proportions, '
            f'identical age and gender. Do not change the person in any way. '
            f'Now place this same person in a new setting and season: {en}. '
            f'Atmosphere: {mood}. '
            f'Wardrobe: {clothes}. '
            f'Background and lighting: {scene_base}. '
            f'Photorealistic cinematic film still, natural skin texture, ultra detailed, sharp focus on the eyes, '
            f'85mm lens, shallow depth of field, 8k uhd, single person, '
            f'no text, no watermark, no logo.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--of', type=int, default=1)
    ap.add_argument('--url', default=None)
    ap.add_argument('--out', default=ROOT)
    args = ap.parse_args()
    if args.url:
        os.environ['COMFY_URL'] = args.url
        importlib.reload(comfy)

    cast = json.load(open(CAST, encoding='utf-8'))
    mine = [c for i, c in enumerate(cast) if i % args.of == args.shard]
    os.makedirs(args.out, exist_ok=True)
    log = open(f'{args.out}/shard{args.shard}.log', 'a', encoding='utf-8')

    def say(*a):
        m = ' '.join(str(x) for x in a)
        print(m, flush=True)
        log.write(m + '\n')
        log.flush()

    say(f'=== shard {args.shard}/{args.of}: {len(mine)} roles x {len(VARIANTS)} variants on {comfy.URL} ===')
    neg = ('lowres, bad anatomy, bad hands, extra fingers, deformed face, extra limbs, two people, crowd, '
           'watermark, text, cartoon, anime, illustration, plastic skin, blurry, low quality')

    for c in mine:
        cid = c['id']
        hero = f'/ssd2/duanju/characters/cast/{cid}/{cid}_hero.png'
        if not os.path.exists(hero):
            say(f'{cid} no hero, skip')
            continue
        cdir = os.path.join(args.out, cid)
        os.makedirs(cdir, exist_ok=True)
        sex_off = 0 if cid[0] == 'M' else 50
        base_seed = 1600000 + sex_off + int(cid[1:]) * 100
        scene_base = c.get('scene_en', 'in a softly lit neutral interior with shallow depth of field')

        for vi, (key, zh, en, mood, clothes) in enumerate(VARIANTS):
            dst = os.path.join(cdir, f'{cid}_{key}.png')
            if os.path.exists(dst) and os.path.getsize(dst) > 20000:
                continue
            p = prompt_for(c, en, mood, clothes, scene_base)
            ok = False
            for attempt in range(3):
                try:
                    fs = GC.run_one(GC.up(hero), p, base_seed + vi * 10 + attempt,
                                    f'{cid}_{key}', cdir, neg=neg)
                    if fs:
                        os.replace(fs[0], dst)
                        ok = True
                        break
                except Exception as ex:
                    say(f'  {cid} {key} attempt {attempt}: {ex}')
                    time.sleep(3)
            if ok:
                say(f'  {cid} {zh}({key}) -> {dst}')
            else:
                say(f'  {cid} {zh}({key}) FAILED')

        say(f'{cid} variants done')

    say('=== shard finished ===')


if __name__ == '__main__':
    main()
