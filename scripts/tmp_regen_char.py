#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重生成指定角色的全部 10 张视图（用于修复撞图/损坏）。

用法：python3 tmp_regen_char.py --ids M12 --url http://127.0.0.1:8190
会先删除该角色已有的所有视图，再重新生成。
"""
import argparse, importlib, json, os, shutil, sys, time

sys.path.insert(0, '/ssd2/short_drama/dh_scripts')
sys.path.insert(0, '/ssd2/duanju/scripts')
import comfy
import gen_cast as GC

CAST = "/ssd2/duanju/scripts/cast_all_150.json"
ROOT = "/ssd2/duanju/characters/cast"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ids', required=True)
    ap.add_argument('--url', required=True)
    ap.add_argument('--seed_off', type=int, default=500000)
    args = ap.parse_args()

    os.environ['COMFY_URL'] = args.url
    importlib.reload(comfy)

    cast = {c['id']: c for c in json.load(open(CAST, encoding='utf-8'))}
    for cid in [x.strip() for x in args.ids.split(',') if x.strip()]:
        c = cast[cid]
        cdir = os.path.join(ROOT, cid)
        # 清空旧图
        for p in os.listdir(cdir) if os.path.isdir(cdir) else []:
            if p.endswith('.png'):
                os.remove(os.path.join(cdir, p))
        os.makedirs(cdir, exist_ok=True)

        scene = c.get('scene_en', '')
        sex_off = 0 if cid[0] == 'M' else 50
        base_seed = 910000 + sex_off + int(cid[1:]) * 100 + args.seed_off

        master = os.path.join(cdir, f'{cid}_hero.png')
        prompt = (f'A photorealistic cinematic master portrait of {c["desc"]}, {scene}, {GC.STYLE}')
        ok = False
        for attempt in range(4):
            try:
                fs = GC.run_one(GC.up('blank:832x1216'), prompt, base_seed + attempt,
                                f'{cid}_hero', cdir,
                                neg='lowres, bad anatomy, bad hands, extra fingers, deformed face, '
                                    'watermark, text, cartoon, anime, illustration, plastic skin')
                if fs:
                    os.replace(fs[0], master)
                    ok = True
                    break
            except Exception as ex:
                print(f'  {cid} hero attempt {attempt}: {ex}', flush=True)
                time.sleep(3)
        if not ok:
            print(f'{cid} HERO FAILED', flush=True)
            continue
        print(f'{cid} hero ok (seed {base_seed})', flush=True)

        keep = ('Keep the exact same person and the exact same face as in image 1, identical facial features, '
                'identical hairstyle, identical skin tone, identical body build')
        views = [
            ('headshot', f'{keep}. Change only the framing: an extreme close-up head-and-shoulders portrait, face filling '
                         f'the frame, neutral expression, looking straight into the camera, plain neutral light grey seamless '
                         f'studio background, soft even beauty lighting, sharp focus on the eyes, natural skin texture, '
                         f'photorealistic, 8k, no text, no watermark.'),
            ('threequarter', f'{keep}, identical clothing, identical {scene}. Change only the camera angle: a three-quarter view, '
                             f'head turned about 40 degrees away from the camera, shoulders slightly rotated, eyes still looking '
                             f'toward the camera, {GC.STYLE}'),
            ('profile', f'{keep}, identical clothing, identical {scene}. Change only the camera angle: a strict side profile '
                        f'view, head turned exactly 90 degrees to the side, showing the clear silhouette of the nose, lips and '
                        f'jawline, eyes looking forward, {GC.STYLE}'),
            ('back', f'{keep}, identical clothing, identical hairstyle and identical body build. Change only the camera '
                     f'angle: a full back view, the back completely facing the camera, the back of the head and the back '
                     f'visible, face not visible, standing upright, same {scene}, {GC.STYLE}'),
            ('fullbody', f'{keep}, identical clothing. Change only the framing: zoom out to a full body shot, standing '
                         f'upright with a natural relaxed posture, whole figure visible from head to toe with clear space '
                         f'above the head and below the feet, same {scene}, {GC.STYLE}'),
            ('smile', f'{keep}, identical clothing, identical background and identical lighting. Change only the '
                      f'expression: a warm natural smile with soft crinkled eyes, same head position facing the camera, '
                      f'{GC.STYLE}'),
            ('angry', f'{keep}, identical clothing, identical background and identical lighting. Change only the '
                  f'expression: furious and cold, brows pulled down and together, jaw clenched, intense glare, same '
                  f'head position facing the camera, {GC.STYLE}'),
            ('sad', f'{keep}, identical clothing, identical background and identical lighting. Change only the '
                    f'expression: deeply sorrowful, glistening eyes with a faint tear, downturned lips, lowered gaze, '
                    f'same head position facing the camera, {GC.STYLE}'),
        ]
        for vi, (vn, vp) in enumerate(views):
            dst = os.path.join(cdir, f'{cid}_{vn}.png')
            for attempt in range(3):
                try:
                    fs = GC.run_one(GC.up(master), vp, base_seed + 10 + vi + attempt * 3,
                                    f'{cid}_{vn}', cdir)
                    if fs:
                        if fs[0] != dst:
                            os.replace(fs[0], dst)
                        break
                except Exception as ex:
                    print(f'  {cid} {vn} attempt {attempt}: {ex}', flush=True)
                    time.sleep(2)

        ta = os.path.join(cdir, f'{cid}_turnaround.png')
        tp = (f'A professional character turnaround reference sheet of the exact same person shown in image 2. '
              f'Three full-body views of the same person standing side by side, evenly spaced, identical in '
              f'scale, on a plain neutral light grey studio background with soft even lighting and a subtle '
              f'floor shadow. Left: front view facing the camera. Middle: 90 degree side profile view. '
              f'Right: back view. All three figures have the identical face, identical hairstyle and identical '
              f'clothing and body proportions. Clean horizontal layout, generous margins, sharp focus on '
              f'every figure, photorealistic, 8k character design sheet, no text, no labels, no watermark.')
        for attempt in range(3):
            try:
                fs = GC.run_one(GC.up('blank:1536x832'), tp, base_seed + 30 + attempt * 3,
                                f'{cid}_turnaround', cdir, img2=GC.up(master))
                if fs:
                    if fs[0] != ta:
                        os.replace(fs[0], ta)
                    break
            except Exception as ex:
                print(f'  {cid} turnaround attempt {attempt}: {ex}', flush=True)
                time.sleep(2)

        n = len([p for p in os.listdir(cdir) if p.endswith('.png')])
        print(f'{cid} done: {n} images', flush=True)


if __name__ == '__main__':
    main()
