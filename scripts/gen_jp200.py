#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 200 张日本氛围背景图（自然环境 + 街景，纯空场景，全程禁文字）。

  * 竖版 832x1472 输入 -> 输出 752x1392（9:16）
  * 7 个 ComfyUI 实例分片并行
  * prompt 强调日本氛围，negative 禁一切文字
  * 已存在且够大的文件跳过，可断点续跑
  * 清理 ComfyUI 多文件残留，避免旧图冒充新图

用法: gen_jp200.py --shard 0 --of 7 --url http://127.0.0.1:8188
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import zlib

sys.path.insert(0, '/ssd2/short_drama/dh_scripts')
import comfy  # noqa: E402

SCENES = '/ssd2/duanju/scripts/scenes_jp200.json'
OUT = '/ssd2/duanju/scenes_jp200'
SIZE_P = '832x1472'    # 竖版 9:16
SIZE_L = '1472x832'    # 横版 16:9
SIZE = SIZE_P          # 兼容旧调用

UNET = 'qwen_image_edit_2511_int8_convrot.safetensors'
CLIP = 'qwen_2.5_vl_7b_fp8_scaled.safetensors'
VAE = 'qwen_image_vae.safetensors'
LORA = 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'

STYLE = ("photorealistic cinematic environment shot of a Japanese scene, authentic Japanese architecture and "
         "landscape, natural lighting, ultra detailed, 8k uhd, architectural photography, wide angle, "
         "sharp focus, depth of field, no people, no human figures, no mannequin, no crowd, "
         "no silhouette of a person")

NEG = ("text, letters, words, characters, chinese characters, japanese kanji, korean hangul, "
       "numbers, digits, symbols, glyphs, runes, typography, caption, subtitle, watermark, logo, "
       "signature, scribble, graffiti, calligraphy, writing, printed words, signage, lettering, "
       "inscription, plaque lettering, banner text, poster text, street art, "
       "people, person, human, crowd, figures, faces, bodies, mannequin, "
       "lowres, bad anatomy, extra limbs, blurry, distorted, cartoon, anime, illustration, painting")


def blank_png(spec):
    import struct
    w, h = map(int, spec.split(':', 1)[1].lower().split('x'))
    raw = b''.join(b'\x00' + b'\x80' * (3 * w) for _ in range(h))

    def chunk(t, d):
        tb = t.encode('ascii') if isinstance(t, str) else t
        return (struct.pack('>I', len(d)) + tb + d
                + struct.pack('>I', zlib.crc32(tb + d) & 0xffffffff))

    path = os.path.join(tempfile.gettempdir(), f'blank_{w}x{h}_{os.getpid()}.png')
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n'
                + chunk('IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                + chunk('IDAT', zlib.compress(raw, 9))
                + chunk('IEND', b''))
    return path


def up(path):
    from PIL import Image
    src = blank_png(path) if path.startswith('blank:') else path
    tmpdir = tempfile.mkdtemp(prefix='cimg_')
    try:
        dst = os.path.join(tmpdir, 'img.jpg')
        with Image.open(src) as im:
            im.convert('RGB').save(dst, quality=96, subsampling=0)
        return comfy.upload(dst)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def build_graph(img, prompt, seed, prefix):
    return {
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
            'clip': ['162', 0], 'vae': ['146', 0], 'image1': ['160', 0], 'prompt': NEG}},
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
            'model': ['153', 0], 'positive': ['148', 0], 'negative': ['147', 0],
            'latent_image': ['156', 0], 'seed': seed, 'steps': 4, 'cfg': 1.0,
            'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}},
        '158': {'class_type': 'VAEDecode', 'inputs': {'samples': ['169', 0], 'vae': ['146', 0]}},
        '195': {'class_type': 'SaveImage', 'inputs': {'images': ['158', 0], 'filename_prefix': prefix}},
    }


def _image_ok(path):
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def run_one(img, prompt, seed, prefix, out_dir):
    e = comfy.run(build_graph(img, prompt, seed, prefix))
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
            got = False
            if os.path.exists(dst) and _image_ok(dst):
                files.append(dst)
                got = True
            if not got:
                tmp = dst + '.dl'
                for attempt in range(3):
                    try:
                        with urllib.request.urlopen(f'{comfy.URL}/view?{q}', timeout=600) as r, \
                                open(tmp, 'wb') as f:
                            while True:
                                chunk = r.read(262144)
                                if not chunk:
                                    break
                                f.write(chunk)
                        if _image_ok(tmp):
                            os.replace(tmp, dst)
                            got = True
                            break
                    except Exception:
                        pass
                    try:
                        if os.path.exists(tmp):
                            os.remove(tmp)
                    except OSError:
                        pass
                    time.sleep(1.5 * (attempt + 1))
            if not got:
                raise IOError(f'output truncated: {base}')
            files.append(dst)
    return files


def build_prompt(sc):
    """场景 prompt；街道含 sign 时额外要求在上部留一块完全空白的招牌/门牌。"""
    p = (f'A photorealistic cinematic background of {sc["en"]}. '
         f'The scene is completely empty of people. '
         f'There is absolutely no writing, no text, no letters, no characters anywhere. '
         f'Signs, plaques, banners, curtains and lanterns are completely blank with no markings. ')
    if sc.get('sign'):
        vert = sc['sign'].get('vertical')
        if vert:
            p += ('In the upper part of the frame there is a plain blank vertical wooden signboard '
                  'hanging, completely empty and unmarked, with absolutely no writing on it. ')
        else:
            p += ('Across the upper part of the frame there is a plain blank horizontal signboard '
                  'and shop plaque, completely empty and unmarked, with absolutely no writing on it. ')
    p += ('No Chinese characters, no Japanese kanji, no Korean hangul, no English letters, '
          'no numbers, no symbols, no calligraphy, no graffiti, no logos anywhere in the frame. '
          f'{STYLE}')
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--of', type=int, default=1)
    ap.add_argument('--url', default=None)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--scenes', default=SCENES)
    ap.add_argument('--only', default='')
    ap.add_argument('--seed_offset', type=int, default=0)
    args = ap.parse_args()

    if args.url:
        os.environ['COMFY_URL'] = args.url
        import importlib
        importlib.reload(comfy)

    with open(args.scenes, encoding='utf-8') as f:
        scenes = json.load(f)['scenes']
    only = set(x for x in args.only.split(',') if x)
    if only:
        scenes = [s for s in scenes if s['sid'] in only]
    else:
        scenes = [s for i, s in enumerate(scenes) if i % args.of == args.shard]

    os.makedirs(args.out, exist_ok=True)
    log = open(f'{args.out}/shard{args.shard}.log', 'a', encoding='utf-8')
    t_start = time.time()

    def say(*a):
        msg = ' '.join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    say(f'=== shard {args.shard}/{args.of}: {len(scenes)} scenes on {comfy.URL} ===')
    for i, sc in enumerate(scenes):
        sid = sc['sid']
        dst = os.path.join(args.out, f'{sid}.png')
        if os.path.exists(dst) and os.path.getsize(dst) > 20000:
            say(f'  {sid} exists, skip')
            continue
        prompt = build_prompt(sc)
        size = SIZE_P if sc.get('orientation', 'portrait') == 'portrait' else SIZE_L
        seed = 700000 + (zlib.crc32(sid.encode()) % 997) * 37 + args.seed_offset
        ok = False
        for attempt in range(3):
            try:
                prefix = f'scene_jp200/s{args.shard}_{sid}_o{args.seed_offset}_a{attempt}'
                fs = run_one(up(f'blank:{size}'), prompt, seed + attempt, prefix, args.out)
                if fs:
                    src = fs[0]
                    if not _image_ok(src):
                        raise IOError(f'output not decodable: {src}')
                    os.replace(src, dst)
                    sub = os.path.join(args.out, 'scene_jp200')
                    if os.path.isdir(sub):
                        for f in os.listdir(sub):
                            if f.startswith(f's{args.shard}_{sid}'):
                                try:
                                    os.remove(os.path.join(sub, f))
                                except OSError:
                                    pass
                    ok = True
                    say(f'  {sid} -> {dst} ({time.time()-t_start:.0f}s)')
                    break
            except Exception as ex:
                say(f'  {sid} attempt {attempt} failed: {ex}')
                time.sleep(3)
        if not ok:
            say(f'  {sid} FAILED')
    say(f'=== shard {args.shard} finished in {time.time()-t_start:.0f}s ===')


if __name__ == '__main__':
    main()
