#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate a 16-character digital-human cast (8 male + 8 female) with full turnarounds.

Two stages per character:
  1. master portrait  - text to image from a blank canvas (Qwen-Image-Edit-2511 as a t2i engine)
  2. derived views    - image edit from the master portrait, so every view keeps the same face

Views per character (10): hero, headshot, threequarter, profile, back, fullbody, turnaround,
                          smile, angry, sad.

usage: gen_cast.py <cast.json> <out_root> [--shard i --of n] [--url http://127.0.0.1:8188]
"""
import json, os, sys, time, argparse, urllib.request, urllib.parse, tempfile, shutil
from PIL import Image

sys.path.insert(0, '/ssd2/short_drama/dh_scripts')
import comfy


def blank_png(spec):
    """Mid-grey PNG of the size in 'blank:WxH', written with the standard library."""
    import struct, zlib, tempfile
    w, h = map(int, spec.split(':', 1)[1].lower().split('x'))
    raw = b''.join(b'\x00' + b'\x80' * (3 * w) for _ in range(h))
    chunk = lambda t, d: struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    path = os.path.join(tempfile.gettempdir(), f'blank_{w}x{h}.png')
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))
    return path


def up(path):
    """Upload a local file to ComfyUI; 'blank:WxH' becomes a mid-grey canvas of that size.

    两个必要处理：
    1) 文件名加唯一后缀 —— 多实例共享 input 目录时同名文件并发上传会互相覆盖。
    2) 统一转成 RGB JPEG —— ComfyUI 新版 LoadImage 走 pyav 的 VideoFromFile 路径，
       该路径按视频帧解码 PNG 会丢掉 alpha 通道，导致下游
       "cannot reshape array of size N into shape (H,W,3)"。
    """
    import uuid, shutil
    if path.startswith('blank:'):
        src = blank_png(path)
    else:
        src = path
    tmpdir = tempfile.mkdtemp(prefix='cimg_')
    try:
        # 统一转 RGB JPEG（去掉 alpha，且 pyav 对 jpeg 解码路径稳定）
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

# english scene for each character id
SCENES = {
    'M01': 'inside a modern high-rise office at dusk with a blurred city skyline behind him, warm rim lighting from the side',
    'M02': 'in a hospital corridor beside a large window with soft diffused daylight, calm clinical ambience',
    'M03': 'in a dim private club room with dark wood panelling and moody low-key lighting',
    'M04': 'in a university library with warm wooden bookshelves and slanting afternoon sunlight',
    'M05': 'in a traditional study lined with Chinese books, lit by a warm amber desk lamp',
    'M06': 'in an underground parking garage with cold white overhead lights and concrete pillars',
    'M07': 'on a rainy city street at night, neon reflections on wet pavement',
    'M08': 'in front of an ancient mountain temple in drifting mist, with old pines and stone steps',
    'F01': 'in a top-floor executive boardroom with floor-to-ceiling windows and a city skyline behind her',
    'F02': 'by the window of a street corner coffee shop with soft afternoon sunlight and green plants',
    'F03': 'in a high-end law firm meeting room with black marble and leather chairs',
    'F04': 'in a luxurious hotel suite with crystal chandeliers and wine-red velvet furnishings',
    'F05': 'on a snowy mountain peak with drifting snow and a vast pale grey landscape',
    'F06': 'in the living room of an old traditional Chinese mansion with rosewood furniture and a porcelain vase of flowers',
    'F07': 'in a bright recording studio with colourful lighting and acoustic foam panels',
    'F08': 'in an open-plan modern office with glass partitions and green plants',
}

# ==== GENDER LOCK (auto-patched) ====
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

STYLE = ('photorealistic, cinematic film still, natural skin texture, ultra detailed, sharp focus on the eyes, '
         '85mm lens, shallow depth of field, 8k uhd, single person, no text, no watermark, no logo')

STUDIO = 'plain neutral light grey seamless studio background, soft even lighting, subtle floor shadow'


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cast')
    ap.add_argument('out_root')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--of', type=int, default=1)
    ap.add_argument('--url', default=None)
    args = ap.parse_args()
    if args.url:
        os.environ['COMFY_URL'] = args.url
        import importlib
        importlib.reload(comfy)

    cast = json.load(open(args.cast, encoding='utf-8'))
    mine = [c for i, c in enumerate(cast) if i % args.of == args.shard]
    out_root = args.out_root
    os.makedirs(out_root, exist_ok=True)
    log_path = os.path.join(out_root, f'shard{args.shard}.log')
    log = open(log_path, 'a', encoding='utf-8')

    def say(*a):
        msg = ' '.join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    say(f'=== shard {args.shard}/{args.of} on {comfy.URL} : {len(mine)} characters ===')

    for idx, c in enumerate(cast):
        if c['id'] not in [m['id'] for m in mine]:
            continue
        cid = c['id']
        cdir = os.path.join(out_root, cid)
        os.makedirs(cdir, exist_ok=True)
        scene = c.get('scene_en') or SCENES.get(cid, 'in a softly lit neutral interior with shallow depth of field')
        # 种子必须全局唯一：M01 与 F01 编号相同，需用性别段区分，避免撞图
        sex_off = 0 if cid[0] == 'M' else 50
        base_seed = 910000 + sex_off + int(cid[1:]) * 100
        t_all = time.time()

        # ---- stage 1: master portrait (text to image) ----
        master = os.path.join(cdir, f'{cid}_hero.png')
        if not os.path.exists(master):
            prompt = (f'A photorealistic cinematic master portrait of {c["desc"]}, {scene}, '
                      f'{STYLE}')
            prompt = gender_lock(c, prompt)
            neg_p = gender_neg(c, 'lowres, bad anatomy, bad hands, extra fingers, deformed face, '
                                     'watermark, text, cartoon, anime, illustration, plastic skin')
            for attempt in range(3):
                try:
                    # prefix 必须带角色 ID：多个 ComfyUI 实例共享同一 output 目录时，
                    # 固定前缀会让并发任务写到同一文件名，互相覆盖导致图片损坏
                    fs = run_one(up('blank:832x1216'), prompt, base_seed, f'{cid}_hero', cdir,
                                 neg=neg_p)
                    if fs:
                        master = os.path.join(cdir, f'{cid}_hero.png')
                        os.replace(fs[0], master)
                        break
                except Exception as ex:
                    say(f'  {cid} hero attempt {attempt} failed: {ex}')
                    time.sleep(3)
            say(f'{cid} hero -> {master} ({time.time()-t_all:.0f}s)')
        else:
            say(f'{cid} hero exists, skip')

        if not os.path.exists(master):
            say(f'{cid} HERO FAILED, skip derivatives')
            continue

        # ---- stage 2: derived views ----
        keep = ('Keep the exact same person and the exact same face as in image 1, identical facial features, '
                'identical hairstyle, identical skin tone, identical body build')

        views = [
            ('headshot', 'portrait',
             f'{keep}. Change only the framing: an extreme close-up head-and-shoulders portrait, face filling '
             f'the frame, neutral expression, looking straight into the camera, plain neutral light grey seamless '
             f'studio background, soft even beauty lighting, sharp focus on the eyes, natural skin texture, '
             f'photorealistic, 8k, no text, no watermark.'),
            ('threequarter', 'portrait',
             f'{keep}, identical clothing, identical {scene}. Change only the camera angle: a three-quarter view, '
             f'head turned about 40 degrees away from the camera, shoulders slightly rotated, eyes still looking '
             f'toward the camera, {STYLE}'),
            ('profile', 'portrait',
             f'{keep}, identical clothing, identical {scene}. Change only the camera angle: a strict side profile '
             f'view, head turned exactly 90 degrees to the side, showing the clear silhouette of the nose, lips and '
             f'jawline, eyes looking forward, {STYLE}'),
            ('back', 'fullbody',
             f'{keep}, identical clothing, identical hairstyle and identical body build. Change only the camera '
             f'angle: a full back view, the back completely facing the camera, the back of the head and the back '
             f'visible, face not visible, standing upright, same {scene}, {STYLE}'),
            ('fullbody', 'fullbody',
             f'{keep}, identical clothing. Change only the framing: zoom out to a full body shot, standing '
             f'upright with a natural relaxed posture, whole figure visible from head to toe with clear space '
             f'above the head and below the feet, same {scene}, {STYLE}'),
            ('smile', 'portrait',
             f'{keep}, identical clothing, identical background and identical lighting. Change only the '
             f'expression: a warm natural smile with soft crinkled eyes, same head position facing the camera, '
             f'{STYLE}'),
            ('angry', 'portrait',
             f'{keep}, identical clothing, identical background and identical lighting. Change only the '
             f'expression: furious and cold, brows pulled down and together, jaw clenched, intense glare, same '
             f'head position facing the camera, {STYLE}'),
            ('sad', 'portrait',
             f'{keep}, identical clothing, identical background and identical lighting. Change only the '
             f'expression: deeply sorrowful, glistening eyes with a faint tear, downturned lips, lowered gaze, '
             f'same head position facing the camera, {STYLE}'),
        ]

        for vi, (vname, _kind, prompt) in enumerate(views):
            dst = os.path.join(cdir, f'{cid}_{vname}.png')
            if os.path.exists(dst):
                say(f'  {cid} {vname} exists, skip')
                continue
            try:
                fs = run_one(up(master), prompt, base_seed + 10 + vi, f'{cid}_{vname}', cdir)
                if fs:
                    dst = os.path.join(cdir, f'{cid}_{vname}.png')
                    if fs[0] != dst:
                        os.replace(fs[0], dst)
                    say(f'  {cid} {vname} -> {dst}')
            except Exception as ex:
                say(f'  {cid} {vname} FAILED: {ex}')

        # turnaround sheet: landscape canvas as target, master portrait as identity reference
        ta = os.path.join(cdir, f'{cid}_turnaround.png')
        if not os.path.exists(ta):
            tp = (f'A professional character turnaround reference sheet of the exact same person shown in image 2. '
                  f'Three full-body views of the same person standing side by side, evenly spaced, identical in '
                  f'scale, on a plain neutral light grey studio background with soft even lighting and a subtle '
                  f'floor shadow. Left: front view facing the camera. Middle: 90 degree side profile view. '
                  f'Right: back view. All three figures have the identical face, identical hairstyle and identical '
                  f'clothing and body proportions. Clean horizontal layout, generous margins, sharp focus on '
                  f'every figure, photorealistic, 8k character design sheet, no text, no labels, no watermark.')
            try:
                fs = run_one(up('blank:1536x832'), tp, base_seed + 30, f'{cid}_turnaround', cdir,
                             img2=up(master))
                if fs:
                    dst = os.path.join(cdir, f'{cid}_turnaround.png')
                    if fs[0] != dst:
                        os.replace(fs[0], dst)
                    say(f'  {cid} turnaround -> {dst}')
            except Exception as ex:
                say(f'  {cid} turnaround FAILED: {ex}')

        say(f'{cid} done in {time.time()-t_all:.0f}s')

    say('=== shard finished ===')


if __name__ == '__main__':
    main()
