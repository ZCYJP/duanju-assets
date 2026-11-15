#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""共享任务池 worker v2：支持 scene / char / season 三种任务类型。

任务领取用原子 rename（pending/ -> work/），多实例天然互斥。
任务格式：
  {"kind":"season","cid":"M01","variant":"lichun","zh":"立春","en":"...","mood":"...","clothes":"...","scene_base":"..."}
"""
import argparse, importlib, json, os, shutil, sys, time

sys.path.insert(0, '/ssd2/short_drama/dh_scripts')
sys.path.insert(0, '/ssd2/duanju/scripts')
import comfy

GEN_CAST = importlib.import_module('gen_cast')
GEN_SCENES = importlib.import_module('gen_scenes_100')

NEG = ('lowres, bad anatomy, bad hands, extra fingers, deformed face, extra limbs, two people, crowd, '
       'watermark, text, cartoon, anime, illustration, plastic skin, blurry, low quality')


def claim(task_dir, prefix='t_'):
    pend = os.path.join(task_dir, 'pending')
    work = os.path.join(task_dir, 'work')
    os.makedirs(work, exist_ok=True)
    for n in sorted(os.listdir(pend)):
        if not n.startswith(prefix):
            continue
        src = os.path.join(pend, n)
        dst = os.path.join(work, n)
        try:
            os.rename(src, dst)
        except OSError:
            continue
        try:
            with open(dst, encoding='utf-8') as f:
                return dst, json.loads(f.read())
        except Exception:
            os.remove(dst)
    return None, None


def release(task_dir, path, ok, note=''):
    dst = os.path.join(task_dir, 'done' if ok else 'failed')
    os.makedirs(dst, exist_ok=True)
    base = os.path.basename(path)
    target = os.path.join(dst, base)
    if ok:
        os.rename(path, target)
    else:
        shutil.move(path, target)
    with open(target + ('.log' if ok else '.err'), 'a', encoding='utf-8') as f:
        f.write(note + '\n')


def do_season(t, out_root, log):
    cid = t['cid']
    key = t['variant']
    hero = f'/ssd2/duanju/characters/cast/{cid}/{cid}_hero.png'
    if not os.path.exists(hero):
        return False, f'{cid} no hero'
    cdir = os.path.join(out_root, cid)
    os.makedirs(cdir, exist_ok=True)
    dst = os.path.join(cdir, f'{cid}_{key}.png')
    if os.path.exists(dst) and os.path.getsize(dst) > 20000:
        return True, f'{cid} {key} exists'
    p = (f'Keep the exact same person and the exact same face as in image 1: identical facial features, '
         f'identical hairstyle and hair colour, identical skin tone, identical body build and proportions, '
         f'identical age and gender. Do not change the person in any way. '
         f'Now place this same person in a new setting and season: {t["en"]}. '
         f'Atmosphere: {t["mood"]}. '
         f'Wardrobe: {t["clothes"]}. '
         f'Background and lighting: {t["scene_base"]}. '
         f'Photorealistic cinematic film still, natural skin texture, ultra detailed, sharp focus on the eyes, '
         f'85mm lens, shallow depth of field, 8k uhd, single person, '
         f'no text, no watermark, no logo.')
    seed = t['seed']
    for attempt in range(3):
        try:
            fs = GEN_CAST.run_one(GEN_CAST.up(hero), p, seed + attempt, f'{cid}_{key}', cdir, neg=NEG)
            if fs:
                os.replace(fs[0], dst)
                return True, f'{cid} {t["zh"]}({key}) -> {dst}'
        except Exception as ex:
            log(f'  {cid} {key} attempt {attempt}: {ex}')
            time.sleep(3)
    return False, f'{cid} {key} FAILED'


def do_scene(t, out_root, log):
    sid = t['sid']
    dst = f"{out_root}/{sid}.png"
    if os.path.exists(dst) and os.path.getsize(dst) > 20000:
        return True, 'exists'
    prompt = (f'A photorealistic cinematic background of {t["en"]}. '
              f'The scene is empty of people. '
              f'There is a prominent clear sign in the scene displaying exactly this text: "{t["text"]}". '
              f'The text is written in {t["lang"]}, every character is correct and legible, '
              f'rendered with sharp clean strokes on a realistic signboard, '
              f'spelled correctly, no misspelled letters, no random symbols. '
              f'{GEN_SCENES.STYLE}')
    for attempt in range(3):
        try:
            fs = GEN_SCENES.run_one(GEN_SCENES.up('blank:1216x832'), prompt, t['seed'] + attempt,
                                    f'scene_{sid}', out_root, GEN_SCENES.NEG)
            if fs:
                os.replace(fs[0], dst)
                return True, f'{sid} -> {dst}'
        except Exception as ex:
            log(f'  {sid} attempt {attempt} failed: {ex}')
            time.sleep(3)
    return False, f'{sid} FAILED'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--tasks', default='/ssd2/duanju/pool')
    ap.add_argument('--scene_out', default='/ssd2/duanju/scenes_100')
    ap.add_argument('--season_out', default='/ssd2/duanju/seasons')
    ap.add_argument('--worker', default='w')
    args = ap.parse_args()

    os.environ['COMFY_URL'] = args.url
    importlib.reload(comfy)
    os.makedirs(args.season_out, exist_ok=True)

    logf = open(f'/tmp/pool_{args.worker}.log', 'a', encoding='utf-8')

    def log(*a):
        m = ' '.join(str(x) for x in a)
        print(m, flush=True)
        logf.write(m + '\n')
        logf.flush()

    log(f'=== worker {args.worker} on {comfy.URL} start ===')
    idle = 0
    while True:
        path, t = claim(args.tasks)
        if t is None:
            idle += 1
            if idle > 40:
                log('pool empty, worker exit')
                break
            time.sleep(6)
            continue
        idle = 0
        kind = t.get('kind')
        try:
            if kind == 'season':
                ok, note = do_season(t, args.season_out, log)
            elif kind == 'scene':
                ok, note = do_scene(t, args.scene_out, log)
            else:
                ok, note = False, f'unknown kind {kind}'
        except Exception as ex:
            ok, note = False, f'{kind} exception: {ex}'
        release(args.tasks, path, ok, note)
        log(f'[{kind}] {note}')
    log('=== worker exit ===')


if __name__ == '__main__':
    main()
