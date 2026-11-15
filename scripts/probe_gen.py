#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe: 竖版短剧背景图出图链路。

验证三件事：
  1. Qwen-Image-Edit-2511 int8 + Lightning LoRA 在竖版 blank canvas 上的输出尺寸
  2. 单张耗时（用于估算全量 100 张的墙钟时间）
  3. 模型会不会自己乱画文字（prompt 要求空白招牌板 + negative 禁文字）

用法: probe_gen.py --url http://127.0.0.1:8188 --out /ssd2/duanju/_probe_b100
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, '/ssd2/short_drama/dh_scripts')
import comfy  # noqa: E402

OUT = '/ssd2/duanju/_probe_b100'
SIZE = '832x1472'

UNET = 'qwen_image_edit_2511_int8_convrot.safetensors'
CLIP = 'qwen_2.5_vl_7b_fp8_scaled.safetensors'
VAE = 'qwen_image_vae.safetensors'
LORA = 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'

STYLE = ("photorealistic cinematic environment shot, natural lighting, ultra detailed, 8k uhd, "
         "architectural photography, wide angle, sharp focus, no people, no human figures, "
         "no mannequin, no crowd")

NEG = ("text, letters, words, characters, chinese characters, japanese kanji, korean hangul, "
       "typography, caption, subtitle, watermark, logo, signature, scribble, calligraphy, "
       "people, person, human, crowd, figures, faces, bodies, mannequin, "
       "lowres, bad anatomy, extra limbs, blurry, distorted, cartoon, anime, illustration, painting")

PROBES = [
    ('probe_city_sign',
     'a rainy neon city street at night with wet asphalt, shopfronts and tall glass towers',
     'a large rectangular shop signboard above the storefront, its face is a completely blank flat surface'),
    ('probe_tea_sign',
     'a traditional chinese tea house interior with bamboo blinds and clay teapots on a wooden table',
     'a horizontal wooden plaque above the counter, its face is a completely blank flat surface'),
    ('probe_ancient_gate',
     'an ancient chinese mountain gate with stone steps, carved stone lions and misty peaks behind',
     'a tall vertical stone tablet beside the gate, its face is a completely blank flat surface'),
    ('probe_susp_office',
     'a dimly lit investigation room at night with a metal desk, scattered case files and a single lamp',
     'a rectangular nameplate on the front of the desk, its face is a completely blank flat surface'),
]


def blank_png(spec):
    import struct
    import tempfile
    import zlib
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
    """上传到 ComfyUI；统一转 RGB JPEG（ComfyUI 新版 LoadImage 会按视频帧解码丢 alpha）。"""
    import shutil
    import tempfile
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
            with urllib.request.urlopen(f'{comfy.URL}/view?{q}', timeout=600) as r, open(dst, 'wb') as f:
                f.write(r.read())
            files.append(dst)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default=None)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--only', default=None, help='只跑某个 probe 名')
    args = ap.parse_args()
    if args.url:
        os.environ['COMFY_URL'] = args.url
        import importlib
        importlib.reload(comfy)

    os.makedirs(args.out, exist_ok=True)
    print(f'=== probe on {comfy.URL} size={SIZE} ===', flush=True)
    for i, (sid, en, sign_desc) in enumerate(PROBES):
        if args.only and args.only != sid:
            continue
        dst = os.path.join(args.out, f'{sid}.png')
        prompt = (f'A photorealistic cinematic background of {en}. '
                  f'The scene is empty of people. '
                  f'There is {sign_desc}, completely blank, absolutely no writing on it. '
                  f'{STYLE}')
        t0 = time.time()
        try:
            fs = run_one(up(f'blank:{SIZE}'), prompt, 880000 + i * 31, f'probe_b100/{sid}', args.out)
            if fs:
                os.replace(fs[0], dst)
                from PIL import Image
                with Image.open(dst) as im:
                    sz = im.size
                # 清理 ComfyUI 多文件残留，避免旧图冒充新图
                leftovers = [f for f in os.listdir(args.out) if sid in f and f != os.path.basename(dst)]
                for f in leftovers:
                    os.remove(os.path.join(args.out, f))
                print(f'OK {sid} {sz} {time.time()-t0:.0f}s -> {dst}', flush=True)
        except Exception as ex:
            print(f'FAIL {sid}: {ex}', flush=True)


if __name__ == '__main__':
    main()
