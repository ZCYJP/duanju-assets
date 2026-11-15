#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 100 种动物角色的形象图（每种 10 视图）。

复用 gen_cast.py 的两阶段链路：
  stage1  master portrait（空白画布 t2i）
  stage2  派生视图（master → img2img，锁同一张脸/同一物种/同一毛色）

与人物的差异：
  * 不做性别锁定（动物无性别硬约束），但 desc 里明确物种与毛色
  * keep 提示词改为「同一角色、同一张脸、同一毛色花纹、同一物种」
  * 视图描述改为动物语义（fur/feathers/scales/muzzle/ears）

用法：
  python3 gen_animals_cast.py /ssd2/duanju/scripts/animals_100.json \
      /ssd2/duanju/characters/animals --shard 0 --of 7 --url http://127.0.0.1:8188
"""
import json, os, sys, time, argparse, urllib.request, urllib.parse, tempfile, shutil
from PIL import Image

sys.path.insert(0, '/ssd2/short_drama/dh_scripts')
import comfy


def blank_png(spec):
    import struct, zlib, tempfile
    w, h = map(int, spec.split(':', 1)[1].lower().split('x'))
    raw = b''.join(b'\x00' + b'\x80' * (3 * w) for _ in range(h))
    chunk = lambda t, d: (struct.pack('>I', len(d)) + t.encode() + d
                          + struct.pack('>I', zlib.crc32(t.encode() + d) & 0xffffffff))
    path = os.path.join(tempfile.gettempdir(), f'blank_{w}x{h}.png')
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk('IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                + chunk('IDAT', zlib.compress(raw, 9)) + chunk('IEND', b''))
    return path


def up(path):
    import uuid
    if path.startswith('blank:'):
        src = blank_png(path)
    else:
        src = path
    tmpdir = tempfile.mkdtemp(prefix='cimg_')
    try:
        dst = os.path.join(tmpdir, 'img.jpg')
        with Image.open(src) as im:
            im.convert('RGB').save(dst, quality=96, subsampling=0)
        return comfy.upload(dst)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


UNET = 'qwen_image_edit_2511_int8_convrot.safetensors'
CLIP = 'qwen_2.5_vl_7b_fp8_scaled.safetensors'
VAE = 'qwen_image_vae.safetensors'
LORA = 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'

STYLE = ('photorealistic cinematic film still, ultra detailed, sharp focus on the eyes, '
         '85mm lens, shallow depth of field, 8k uhd, single character, no text, no watermark, no logo')

NEG = ('lowres, bad anatomy, bad hands, extra fingers, extra limbs, deformed face, deformed body, '
       'watermark, text, caption, signature, cartoon, anime, illustration, plastic skin, '
       'human face, human skin, clothing text')


def build_graph(img, prompt, seed, prefix, neg='', turbo=True, steps=None, cfg=None,
                denoise=1.0, img2=None):
    steps = steps or (4 if turbo else 25)
    cfg = cfg if cfg is not None else (1.0 if turbo else 4.0)
    g = {
        '41': {'class_type': 'LoadImage', 'inputs': {'image': img}},
        '161': {'class_type': 'UNETLoader', 'inputs': {
            'unet_name': comfy.find_model('diffusion_models', UNET), 'weight_dtype': 'default'}},
        '162': {'class_type': 'CLIPLoader', 'inputs': {
            'clip_name': comfy.find_model('text_encoders', CLIP), 'type': 'qwen_image', 'device': 'default'}},
        '146': {'class_type': 'VAELoader', 'inputs': {'vae_name': comfy.find_model('vae', VAE)}},
        '160': {'class_type': 'FluxKontextImageScale', 'inputs': {'image': ['41', 0]}},
        '151': {'class_type': 'TextEncodeQwenImageEditPlus', 'inputs': {
            'clip': ['162', 0], 'vae': ['146', 0], 'image1': ['160', 0], 'prompt': prompt}},
        '149': {'class_type': 'TextEncodeQwenImageEditPlus', 'inputs': {
            'clip': ['162', 0], 'vae': ['146', 0], 'image1': ['160', 0], 'prompt': neg}},
        '148': {'class_type': 'FluxKontextMultiReferenceLatentMethod', 'inputs': {
            'conditioning': ['151', 0], 'reference_latents_method': 'index_timestep_zero'}},
        '147': {'class_type': 'FluxKontextMultiReferenceLatentMethod', 'inputs': {
            'conditioning': ['149', 0], 'reference_latents_method': 'index_timestep_zero'}},
        '145': {'class_type': 'ModelSamplingAuraFlow', 'inputs': {'model': ['161', 0], 'shift': 3.1}},
        '152': {'class_type': 'CFGNorm', 'inputs': {'model': ['145', 0], 'strength': 1.0}},
        '153': {'class_type': 'LoraLoaderModelOnly', 'inputs': {
            'model': ['152', 0], 'lora_name': comfy.find_model('loras', LORA), 'strength_model': 1.0}},
        '156': {'class_type': 'VAEEncode', 'inputs': {'pixels': ['160', 0], 'vae': ['146', 0]}},
        '169': {'class_type': 'KSampler', 'inputs': {
            'model': ['153', 0] if turbo else ['152', 0], 'positive': ['148', 0], 'negative': ['147', 0],
            'latent_image': ['156', 0], 'seed': seed, 'steps': steps, 'cfg': cfg, 'sampler_name': 'euler',
            'scheduler': 'simple', 'denoise': denoise}},
        '158': {'class_type': 'VAEDecode', 'inputs': {'samples': ['169', 0], 'vae': ['146', 0]}},
        '195': {'class_type': 'SaveImage', 'inputs': {'images': ['158', 0], 'filename_prefix': prefix}},
    }
    if not turbo:
        del g['153']
    if img2:
        g['42'] = {'class_type': 'LoadImage', 'inputs': {'image': img2}}
        g['163'] = {'class_type': 'FluxKontextImageScale', 'inputs': {'image': ['42', 0]}}
        g['151']['inputs']['image2'] = ['163', 0]
        g['149']['inputs']['image2'] = ['163', 0]
    return g


def run_one(img, prompt, seed, prefix, out_dir, neg='', turbo=True, steps=None, cfg=None,
            denoise=1.0, img2=None):
    g = build_graph(img, prompt, seed, prefix, neg, turbo, steps, cfg, denoise, img2)
    e = comfy.run(g)
    err = comfy.error(e)
    if err:
        raise RuntimeError(err)
    files = []
    for o in e.get('outputs', {}).values():
        for it in o.get('images', []):
            if it.get('type', 'output') != 'output':
                continue
            q = urllib.parse.urlencode({'filename': it['filename'], 'subfolder': it.get('subfolder', ''),
                                        'type': 'output'})
            base = it['filename']
            if it.get('subfolder'):
                base = os.path.join(it['subfolder'], it['filename'])
            dst = os.path.join(out_dir, base)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with urllib.request.urlopen(f"{comfy.URL}/view?{q}", timeout=600) as r, open(dst, 'wb') as f:
                f.write(r.read())
            files.append(dst)
    return files


def image_ok(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return os.path.getsize(path) > 1024
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('animals')
    ap.add_argument('out_root')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--of', type=int, default=1)
    ap.add_argument('--url', default=None)
    ap.add_argument('--only', default=None, help='逗号分隔的 id 白名单（样片用）')
    args = ap.parse_args()
    if args.url:
        os.environ['COMFY_URL'] = args.url
        import importlib
        importlib.reload(comfy)

    raw = json.load(open(args.animals, encoding='utf-8'))
    animals = raw['animals'] if isinstance(raw, dict) else raw
    if args.only:
        keep = set(x.strip() for x in args.only.split(',') if x.strip())
        animals = [a for a in animals if a['id'] in keep]
    mine = [a for i, a in enumerate(animals) if i % args.of == args.shard]
    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)
    log_path = os.path.join(out_root, f'shard{args.shard}.log')
    log = open(log_path, 'a', encoding='utf-8')

    def say(*a):
        msg = ' '.join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    say(f'=== shard {args.shard}/{args.of} on {comfy.URL} : {len(mine)} animals ===')

    for a in mine:
        cid = a['id']
        cdir = os.path.join(out_root, cid)
        os.makedirs(cdir, exist_ok=True)
        scene = a.get('scene_en') or 'plain neutral light grey seamless studio background, soft even lighting'
        base_seed = 920000 + int(cid[1:]) * 100
        t_all = time.time()

        # ---- stage 1: master portrait ----
        master = os.path.join(cdir, f'{cid}_hero.png')
        if not (os.path.exists(master) and image_ok(master)):
            prompt = (f'A photorealistic cinematic full-body character portrait of {a["desc"]}, '
                      f'{scene}, {STYLE}')
            for attempt in range(3):
                try:
                    fs = run_one(up('blank:832x1216'), prompt, base_seed + attempt,
                                 f'{cid}_hero_o{attempt}', cdir, neg=NEG)
                    if fs and image_ok(fs[0]):
                        os.replace(fs[0], master)
                        break
                except Exception as ex:
                    say(f'  {cid} hero attempt {attempt} failed: {ex}')
                    time.sleep(3)
            say(f'{cid} hero -> {master} ({time.time()-t_all:.0f}s)')
        else:
            say(f'{cid} hero exists, skip')

        if not (os.path.exists(master) and image_ok(master)):
            say(f'{cid} HERO FAILED, skip derivatives')
            continue

        keep = ('Keep the exact same character and the exact same face as in image 1, '
                'identical species, identical fur or feather colour and markings, '
                'identical eye colour, identical body build and identical proportions')

        views = [
            ('headshot', 'portrait',
             f'{keep}. Change only the framing: an extreme close-up head-and-shoulders portrait, '
             f'the face filling the frame, neutral expression, looking straight into the camera, '
             f'plain neutral light grey seamless studio background, soft even lighting, '
             f'sharp focus on the eyes, photorealistic, 8k, no text, no watermark.'),
            ('threequarter', 'portrait',
             f'{keep}, identical outfit, identical {scene}. Change only the camera angle: '
             f'a three-quarter view, head turned about 40 degrees away from the camera, '
             f'shoulders slightly rotated, eyes still looking toward the camera, {STYLE}'),
            ('profile', 'portrait',
             f'{keep}, identical outfit, identical {scene}. Change only the camera angle: '
             f'a strict side profile view, head turned exactly 90 degrees to the side, '
             f'showing the clear silhouette of the muzzle, nose and jawline, eyes looking forward, {STYLE}'),
            ('back', 'fullbody',
             f'{keep}, identical outfit, identical fur markings and identical body build. '
             f'Change only the camera angle: a full back view, the back completely facing the camera, '
             f'the back of the head and the back visible, face not visible, standing upright, '
             f'same {scene}, {STYLE}'),
            ('fullbody', 'fullbody',
             f'{keep}, identical outfit. Change only the framing: zoom out to a full body shot, '
             f'standing upright on two legs with a natural relaxed posture, whole figure visible '
             f'from head to toe with clear space above the head and below the feet, same {scene}, {STYLE}'),
            ('smile', 'portrait',
             f'{keep}, identical outfit, identical background and identical lighting. Change only the '
             f'expression: a warm natural happy smile with soft crinkled eyes, same head position '
             f'facing the camera, {STYLE}'),
            ('angry', 'portrait',
             f'{keep}, identical outfit, identical background and identical lighting. Change only the '
             f'expression: furious and cold, brows pulled down and together, jaw clenched, '
             f'intense glare, same head position facing the camera, {STYLE}'),
            ('sad', 'portrait',
             f'{keep}, identical outfit, identical background and identical lighting. Change only the '
             f'expression: deeply sorrowful, glistening eyes with a faint tear, downturned mouth, '
             f'lowered gaze, same head position facing the camera, {STYLE}'),
        ]

        for vi, (vname, _kind, prompt) in enumerate(views):
            dst = os.path.join(cdir, f'{cid}_{vname}.png')
            if os.path.exists(dst) and image_ok(dst):
                say(f'  {cid} {vname} exists, skip')
                continue
            for attempt in range(2):
                try:
                    fs = run_one(up(master), prompt, base_seed + 10 + vi + attempt * 3,
                                 f'{cid}_{vname}_o{attempt}', cdir, neg=NEG)
                    if fs and image_ok(fs[0]):
                        os.replace(fs[0], dst)
                        say(f'  {cid} {vname} -> {dst}')
                        break
                except Exception as ex:
                    say(f'  {cid} {vname} attempt {attempt} FAILED: {ex}')
                    time.sleep(3)

        # turnaround sheet
        ta = os.path.join(cdir, f'{cid}_turnaround.png')
        if not (os.path.exists(ta) and image_ok(ta)):
            tp = (f'A professional character turnaround reference sheet of the exact same animal '
                  f'character shown in image 2. Three full-body views of the same character standing '
                  f'side by side, evenly spaced, identical in scale, on a plain neutral light grey '
                  f'studio background with soft even lighting and a subtle floor shadow. '
                  f'Left: front view facing the camera. Middle: 90 degree side profile view. '
                  f'Right: back view. All three figures have the identical face, identical fur or '
                  f'feather colour and markings, identical outfit and identical body proportions. '
                  f'Clean horizontal layout, generous margins, sharp focus on every figure, '
                  f'photorealistic, 8k character design sheet, no text, no labels, no watermark.')
            for attempt in range(2):
                try:
                    fs = run_one(up('blank:1536x832'), tp, base_seed + 30 + attempt,
                                 f'{cid}_turnaround_o{attempt}', cdir, neg=NEG, img2=up(master))
                    if fs and image_ok(fs[0]):
                        os.replace(fs[0], ta)
                        say(f'  {cid} turnaround -> {ta}')
                        break
                except Exception as ex:
                    say(f'  {cid} turnaround attempt {attempt} FAILED: {ex}')
                    time.sleep(3)

        say(f'{cid} done in {time.time()-t_all:.0f}s')

    say('=== shard finished ===')


if __name__ == '__main__':
    main()
