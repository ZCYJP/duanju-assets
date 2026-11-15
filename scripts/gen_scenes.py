#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generate the 5 empty-scene reference images for the char16 drama and stage all refs into ComfyUI input.

usage: gen_scenes.py
"""
import json, os, sys, time, urllib.request, urllib.parse

sys.path.insert(0, '/ssd2/short_drama/dh_scripts')
import comfy

UNET = 'qwen_image_edit_2511_int8_convrot.safetensors'
CLIP = 'qwen_2.5_vl_7b_fp8_scaled.safetensors'
VAE = 'qwen_image_vae.safetensors'
LORA = 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'

OUT_DIR = '/ssd2/short_drama/char16/drama/scenes'
STAGE = '/ssd2/tools/ComfyUI/input/char16_drama/scenes'

SCENES = [
    ('scene_boardroom', '832x1216',
     'A completely empty luxurious executive boardroom at night, no people, a long polished dark wood conference '
     'table reflecting soft overhead light, high-backed black leather chairs, floor-to-ceiling windows revealing a '
     'blurred city skyline, warm amber rim light from one side, moody cinematic lighting, ultra detailed '
     'interior photography, 8k, no text'),
    ('scene_office', '832x1216',
     'A completely empty high-rise corner office at dusk, no people, floor-to-ceiling glass windows with a vast '
     'blurred city skyline, dark wood desk, a single leather chair, cold blue evening light outside with warm '
     'interior lamps, deep shadows, cinematic interior photography, 8k, no text'),
    ('scene_garage', '832x1216',
     'A completely empty underground parking garage, no people, rows of concrete pillars, cold white overhead '
     'fluorescent lights, painted lane markings on the polished concrete floor, moody industrial atmosphere, '
     'cinematic wide shot, 8k, no text'),
    ('scene_street', '832x1216',
     'A completely empty city street at night in the pouring rain, no people, wet asphalt reflecting vivid neon '
     'signs in magenta and cyan, heavy rain streaks catching the light, tall glass office buildings on both '
     'sides, cinematic noir atmosphere, shallow depth of field, 8k, no text'),
    ('scene_topfloor', '1536x832',
     'A completely empty top floor corner of a skyscraper at night, no people, panoramic floor-to-ceiling glass '
     'walls overlooking an endless glittering city skyline far below, dark polished floor, minimal furniture, warm '
     'golden interior light contrasting with the cold blue night outside, cinematic wide shot, 8k, no text'),
]


def blank_png(spec):
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
    """Upload to ComfyUI; 'blank:WxH' becomes a mid-grey canvas of that size."""
    return comfy.upload(blank_png(path) if path.startswith('blank:') else path)


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
            'clip': ['162', 0], 'vae': ['146', 0], 'image1': ['160', 0],
            'prompt': 'people, person, crowd, human, mannequin, figures, text, watermark, logo'}},
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
            'latent_image': ['156', 0], 'seed': seed, 'steps': 4, 'cfg': 1.0, 'sampler_name': 'euler',
            'scheduler': 'simple', 'denoise': 1.0}},
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
            dst = os.path.join(out_dir, it['filename'])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with urllib.request.urlopen(f'{comfy.URL}/view?{q}', timeout=600) as r, open(dst, 'wb') as f:
                f.write(r.read())
            files.append(dst)
    return files


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(STAGE, exist_ok=True)
    for i, (name, size, prompt) in enumerate(SCENES):
        dst = os.path.join(OUT_DIR, f'{name}.png')
        if os.path.exists(dst):
            print('exists', dst, flush=True)
            continue
        t0 = time.time()
        try:
            fs = run_one(up(f'blank:{size}'), prompt, 770000 + i * 17, f'scenes/{name}', OUT_DIR)
            if fs:
                os.replace(fs[0], dst)
                print(f'OK {name} -> {dst} ({time.time()-t0:.0f}s)', flush=True)
        except Exception as ex:
            print(f'FAIL {name}: {ex}', flush=True)
    print('=== staged ===')
    for f in sorted(os.listdir(OUT_DIR)):
        print(' ', f)
