#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 100 张短剧背景图（不含任何文字，汉字一律后期用真实字体合成）。

  * 竖版 832x1472 输入 -> 输出 752x1392（9:16）
  * 7 个 ComfyUI 实例分片并行
  * prompt 明确要求空招牌/无字，negative 禁一切文字
  * 已存在且够大的文件跳过，可断点续跑
  * 清理 ComfyUI 多文件残留，避免旧图冒充新图

用法: gen_bg100.py --shard 0 --of 7 --url http://127.0.0.1:8188
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

SCENES = '/ssd2/duanju/scripts/scenes_b100.json'
OUT = '/ssd2/duanju/scenes_b100'
SIZE = '832x1472'

UNET = 'qwen_image_edit_2511_int8_convrot.safetensors'
CLIP = 'qwen_2.5_vl_7b_fp8_scaled.safetensors'
VAE = 'qwen_image_vae.safetensors'
LORA = 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'

STYLE = ("photorealistic cinematic environment shot, natural lighting, ultra detailed, 8k uhd, "
         "architectural photography, wide angle, sharp focus, depth of field, "
         "no people, no human figures, no mannequin, no crowd, no silhouette of a person")

NEG = ("text, letters, words, characters, chinese characters, japanese kanji, korean hangul, "
       "numbers, digits, symbols, glyphs, runes, typography, caption, subtitle, watermark, logo, "
       "signature, scribble, graffiti, calligraphy, writing, printed words, signage, lettering, "
       "inscription, plaque lettering, banner text, poster text, street art, "
       "people, person, human, crowd, figures, faces, bodies, mannequin, "
       "lowres, bad anatomy, extra limbs, blurry, distorted, cartoon, anime, illustration, painting")


def blank_png(spec):
    import struct
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
    """上传到 ComfyUI；统一转 RGB JPEG（新版 LoadImage 按视频帧解码会丢 alpha）。"""
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
    """校验文件是完整可解码的图片（PNG 截断时 PIL 会抛异常）。"""
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

            # 2026-11-15: 原实现一次性 r.read()，服务端分块传输在 ~128/192KB 处
            # 被截断 -> IncompleteRead，整张报废（10/10 全挂）。
            # 修：分块流式读取 + 落盘后完整性校验 + 失败直接取 ComfyUI 本地目录
            #     （ComfyUI 的 --output-directory 就是 out_dir，产物已在服务端落盘，
            #      根本不需要走网络下载）。
            got = False
            # 优先直接用服务端已落盘的产物：ComfyUI 的 --output-directory 就是
            # out_dir，产物在服务端已完整写好，走 /view 下载反而会被截断
            # （且下载路径与服务端路径相同，会把好文件覆盖成半截）。
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
        prompt = (f'A photorealistic cinematic background of {sc["en"]}. '
                  f'The scene is completely empty of people. '
                  f'There is absolutely no writing, no text, no letters, no signs of characters anywhere. '
                  f'Signs, plaques, banners and posters are completely blank with no markings. '
                  f'No Chinese characters, no Japanese kanji, no Korean hangul, no English letters, '
                  f'no numbers, no symbols, no calligraphy, no graffiti, no logos anywhere in the frame. '
                  f'{STYLE}')
        seed = 900000 + (zlib.crc32(sid.encode()) % 997) * 37 + args.seed_offset
        ok = False
        for attempt in range(3):
            try:
                # prefix 必须每次唯一：ComfyUI 的 filename_prefix 计数器是按
                # (subfolder, prefix) 记录的，重复 prefix 会写同名文件，
                # 后一次执行会把前一次的好文件覆盖成 0 字节（2026-11-15 实测）。
                prefix = f'scene_b100/s{args.shard}_{sid}_o{args.seed_offset}_a{attempt}'
                fs = run_one(up(f'blank:{SIZE}'), prompt, seed + attempt, prefix, args.out)
                if fs:
                    src = fs[0]
                    # 落盘前必须校验完整性：ComfyUI 偶发写 0 字节/半截文件
                    if not _image_ok(src):
                        raise IOError(f'output not decodable: {src} '
                                      f'(size={os.path.getsize(src) if os.path.exists(src) else -1})')
                    os.replace(src, dst)
                    # 清理同 sid 的其他残留（避免旧图冒充新图）
                    sub = os.path.join(args.out, 'scene_b100')
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
