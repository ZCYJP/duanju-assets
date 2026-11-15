#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""共享任务池 worker：多个 ComfyUI 实例共同消费同一批任务，避免手动分片负载不均。

用法：
  python3 tmp_pool_worker.py --url http://127.0.0.1:8190 --tasks /ssd2/duanju/pool/tasks.jsonl

任务格式（每行一个 JSON）：
  {"kind":"scene","sid":"bg_xxx","text":"国营第三机械厂","lang":"简体中文","seed":770000}
  {"kind":"char","cid":"M26","cast":{...}}   # char 任务内部会连续完成 hero + 9 视图 + turnaround

任务领取用原子 rename（从 pending/ 移到 work/），天然互斥、崩溃可恢复。
"""
import argparse, json, os, sys, time, shutil

sys.path.insert(0, '/ssd2/short_drama/dh_scripts')
sys.path.insert(0, '/ssd2/duanju/scripts')
import comfy
import importlib

GEN_CAST = importlib.import_module('gen_cast')
GEN_SCENES = importlib.import_module('gen_scenes_100')


def claim(task_dir, prefix):
    """从 pending 目录原子领取一个任务文件；返回 (path, task) 或 (None, None)。"""
    pend = os.path.join(task_dir, 'pending')
    work = os.path.join(task_dir, 'work')
    os.makedirs(work, exist_ok=True)
    names = sorted(os.listdir(pend))
    for n in names:
        if not n.startswith(prefix):
            continue
        src = os.path.join(pend, n)
        dst = os.path.join(work, n)
        try:
            os.rename(src, dst)   # 原子，赢家独占
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
        with open(target + '.log', 'a', encoding='utf-8') as f:
            f.write(note + '\n')
    else:
        shutil.move(path, target)
        with open(target + '.err', 'a', encoding='utf-8') as f:
            f.write(note + '\n')


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


def do_char(t, out_root, log):
    """生成一个角色的全部视图（hero + 8 视图 + turnaround），复用 gen_cast 的逻辑。"""
    c = t['cast']
    cid = c['id']
    cdir = os.path.join(out_root, cid)
    os.makedirs(cdir, exist_ok=True)
    scene = c.get('scene_en') or 'in a softly lit neutral interior with shallow depth of field'
    sex_off = 0 if cid[0] == 'M' else 50
    base_seed = 910000 + sex_off + int(cid[1:]) * 100

    master = os.path.join(cdir, f'{cid}_hero.png')
    if not os.path.exists(master):
        prompt = (f'A photorealistic cinematic master portrait of {c["desc"]}, {scene}, '
                  f'{GEN_CAST.STYLE}')
        for attempt in range(3):
            try:
                fs = GEN_CAST.run_one(GEN_CAST.up('blank:832x1216'), prompt, base_seed,
                                      f'{cid}_hero', cdir,
                                      neg='lowres, bad anatomy, bad hands, extra fingers, deformed face, '
                                          'watermark, text, cartoon, anime, illustration, plastic skin')
                if fs:
                    os.replace(fs[0], master)
                    break
            except Exception as ex:
                log(f'  {cid} hero attempt {attempt} failed: {ex}')
                time.sleep(3)
    if not os.path.exists(master):
        return False, f'{cid} HERO FAILED'

    keep = ('Keep the exact same person and the exact same face as in image 1, identical facial features, '
            'identical hairstyle, identical skin tone, identical body build')
    views = [
        ('headshot', f'{keep}. Change only the framing: an extreme close-up head-and-shoulders portrait, face filling '
                     f'the frame, neutral expression, looking straight into the camera, plain neutral light grey seamless '
                     f'studio background, soft even beauty lighting, sharp focus on the eyes, natural skin texture, '
                     f'photorealistic, 8k, no text, no watermark.'),
        ('threequarter', f'{keep}, identical clothing, identical {scene}. Change only the camera angle: a three-quarter view, '
                         f'head turned about 40 degrees away from the camera, shoulders slightly rotated, eyes still looking '
                         f'toward the camera, {GEN_CAST.STYLE}'),
        ('profile', f'{keep}, identical clothing, identical {scene}. Change only the camera angle: a strict side profile '
                    f'view, head turned exactly 90 degrees to the side, showing the clear silhouette of the nose, lips and '
                    f'jawline, eyes looking forward, {GEN_CAST.STYLE}'),
        ('back', f'{keep}, identical clothing, identical hairstyle and identical body build. Change only the camera '
                 f'angle: a full back view, the back completely facing the camera, the back of the head and the back '
                 f'visible, face not visible, standing upright, same {scene}, {GEN_CAST.STYLE}'),
        ('fullbody', f'{keep}, identical clothing. Change only the framing: zoom out to a full body shot, standing '
                     f'upright with a natural relaxed posture, whole figure visible from head to toe with clear space '
                     f'above the head and below the feet, same {scene}, {GEN_CAST.STYLE}'),
        ('smile', f'{keep}, identical clothing, identical background and identical lighting. Change only the '
                  f'expression: a warm natural smile with soft crinkled eyes, same head position facing the camera, '
                  f'{GEN_CAST.STYLE}'),
        ('angry', f'{keep}, identical clothing, identical background and identical lighting. Change only the '
                  f'expression: furious and cold, brows pulled down and together, jaw clenched, intense glare, same '
                  f'head position facing the camera, {GEN_CAST.STYLE}'),
        ('sad', f'{keep}, identical clothing, identical background and identical lighting. Change only the '
                f'expression: deeply sorrowful, glistening eyes with a faint tear, downturned lips, lowered gaze, '
                f'same head position facing the camera, {GEN_CAST.STYLE}'),
    ]
    for vi, (vname, prompt) in enumerate(views):
        dst = os.path.join(cdir, f'{cid}_{vname}.png')
        if os.path.exists(dst):
            continue
        for attempt in range(2):
            try:
                fs = GEN_CAST.run_one(GEN_CAST.up(master), prompt, base_seed + 10 + vi,
                                      f'{cid}_{vname}', cdir)
                if fs:
                    if fs[0] != dst:
                        os.replace(fs[0], dst)
                    break
            except Exception as ex:
                log(f'  {cid} {vname} attempt {attempt} failed: {ex}')
                time.sleep(2)

    ta = os.path.join(cdir, f'{cid}_turnaround.png')
    if not os.path.exists(ta):
        tp = (f'A professional character turnaround reference sheet of the exact same person shown in image 2. '
              f'Three full-body views of the same person standing side by side, evenly spaced, identical in '
              f'scale, on a plain neutral light grey studio background with soft even lighting and a subtle '
              f'floor shadow. Left: front view facing the camera. Middle: 90 degree side profile view. '
              f'Right: back view. All three figures have the identical face, identical hairstyle and identical '
              f'clothing and body proportions. Clean horizontal layout, generous margins, sharp focus on '
              f'every figure, photorealistic, 8k character design sheet, no text, no labels, no watermark.')
        for attempt in range(2):
            try:
                fs = GEN_CAST.run_one(GEN_CAST.up('blank:1536x832'), tp, base_seed + 30,
                                      f'{cid}_turnaround', cdir, img2=GEN_CAST.up(master))
                if fs:
                    dst = os.path.join(cdir, f'{cid}_turnaround.png')
                    if fs[0] != dst:
                        os.replace(fs[0], dst)
                    break
            except Exception as ex:
                log(f'  {cid} turnaround attempt {attempt} failed: {ex}')
                time.sleep(2)

    need = [f'{cid}_hero.png'] + [f'{cid}_{v}.png' for v, _ in views] + [f'{cid}_turnaround.png']
    miss = [n for n in need if not os.path.exists(os.path.join(cdir, n))]
    if miss:
        return False, f'{cid} missing {miss}'
    return True, f'{cid} complete (10 views)'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--tasks', default='/ssd2/duanju/pool')
    ap.add_argument('--scene_out', default='/ssd2/duanju/scenes_100')
    ap.add_argument('--char_out', default='/ssd2/duanju/characters/cast')
    ap.add_argument('--worker', default='w')
    args = ap.parse_args()

    os.environ['COMFY_URL'] = args.url
    importlib.reload(comfy)
    os.makedirs(args.scene_out, exist_ok=True)
    os.makedirs(args.char_out, exist_ok=True)

    logf = open(f'/tmp/pool_{args.worker}.log', 'a', encoding='utf-8')

    def log(*a):
        m = ' '.join(str(x) for x in a)
        print(m, flush=True)
        logf.write(m + '\n')
        logf.flush()

    log(f'=== worker {args.worker} on {comfy.URL} start ===')
    idle = 0
    while True:
        path, t = claim(args.tasks, 't_')
        if t is None:
            idle += 1
            if idle > 60:
                log('pool empty, worker exit')
                break
            time.sleep(5)
            continue
        idle = 0
        kind = t.get('kind')
        try:
            if kind == 'scene':
                ok, note = do_scene(t, args.scene_out, log)
            elif kind == 'char':
                ok, note = do_char(t, args.char_out, log)
            else:
                ok, note = False, f'unknown kind {kind}'
        except Exception as ex:
            ok, note = False, f'{kind} exception: {ex}'
        release(args.tasks, path, ok, note)
        log(f'[{kind}] {note}')
    log('=== worker exit ===')


if __name__ == '__main__':
    main()
