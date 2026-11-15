#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《零点四十七单》— 底图合成（角色参考图 + 场景图 → 保脸人物底图）

管线：Qwen-Image-Edit-2511 int8_convrot + Lightning 4步 LoRA
输入：characters/cast/<CID>/<CID>_<view>.png（身份锚）
      scenes_100/bg_*.png（环境锚，1248x832 横版 → 先裁竖版 832x1248）
输出：frames/{shot}.png  832x1248 竖版

保脸要点：
  - 双图输入：图1=角色(身份)，图2=场景(环境)，prompt 显式声明"图1人物+图2场景"
  - Continuity Lock 英文硬短语原样嵌入
  - Lightning 4步 LoRA + CFG 1.0
  - 场景横转竖：中心裁切优先，宽度不足时上下补暗

用法：gen_frames.py [shot_name ...]   不带参数 = 全部
"""
import os
import sys
import json
import time
import uuid
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from story import SHOTS, LOCK, SCENES, SCENE_DESC, NEGATIVE, CAST_ROOT, BASE

COMFY_PORTS = [8188, 8189, 8190, 8191, 8192, 8193, 8194]
FRAMES = f"{BASE}/frames"
WORK = f"{BASE}/work"
LOGS = f"{BASE}/logs"
os.makedirs(FRAMES, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

UNET = "qwen_image_edit_2511_int8_convrot.safetensors"
CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

W, H = 832, 1248
STEPS = 4
CFG = 1.0

OUT_ROOT = {8188: "/ssd2/duanju/comfy_out_8188", 8189: "/ssd2/duanju/comfy_out_8189",
            8190: "/ssd2/duanju/comfy_out_8190", 8191: "/ssd2/duanju/comfy_out_8191",
            8192: "/ssd2/duanju/comfy_out_8192", 8193: "/ssd2/duanju/comfy_out_8193",
            8194: "/ssd2/duanju/comfy_out_8194"}


def node(cls, **kw):
    return {"class_type": cls, "inputs": kw}


def upload_image(port, path):
    boundary = "----gf47" + uuid.uuid4().hex
    name = os.path.basename(path)
    data = open(path, "rb").read()
    body = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'
            f"Content-Type: image/png\r\n\r\n").encode() + data + \
           f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def post(url, payload, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get_json(url, timeout=30):
    """GET 请求。ComfyUI 的 /history/{pid} 只接受 GET，用 POST 会返回 405。"""
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def portrait_scene(src, dst):
    """横版场景图 → 竖版 832x1248。优先中心裁切，宽度不足则左右模糊边填充。"""
    from PIL import Image, ImageFilter, ImageEnhance
    if os.path.exists(dst) and os.path.getsize(dst) > 20000:
        return dst
    im = Image.open(src).convert("RGB")
    sw, sh = im.size
    target_ratio = W / H
    if sw / sh > target_ratio:
        # 太宽：中心裁切
        nw = int(sh * target_ratio)
        left = (sw - nw) // 2
        im = im.crop((left, 0, left + nw, sh))
    else:
        # 太高：中心裁切上下
        nh = int(sw / target_ratio)
        top = (sh - nh) // 2
        im = im.crop((0, top, sw, top + nh))
    im = im.resize((W, H), Image.LANCZOS)
    # 轻微提暗，压掉过曝区域，避免底图抢戏
    im = ImageEnhance.Brightness(im).enhance(0.94)
    im = ImageEnhance.Contrast(im).enhance(1.03)
    im.save(dst)
    return dst


def build_workflow_lora(char_ref, scene_ref, prompt, seed, shot):
    wf = {
        "1": node("UNETLoader", unet_name=UNET, weight_dtype="default"),
        "2": node("CLIPLoader", clip_name=CLIP, type="qwen_image", device="default"),
        "3": node("VAELoader", vae_name=VAE),
        "4": node("LoraLoaderModelOnly", lora_name=LORA, strength_model=1.0, model=["1", 0]),
        "5": node("ModelSamplingAuraFlow", shift=3.1, model=["4", 0]),
        "6": node("LoadImage", image=char_ref),
        "7": node("LoadImage", image=scene_ref),
        "8": node("TextEncodeQwenImageEditPlus", clip=["2", 0], prompt=prompt,
                  image1=["6", 0], image2=["7", 0], vae=["3", 0]),
        "9": node("TextEncodeQwenImageEditPlus", clip=["2", 0], prompt="",
                  image1=["7", 0], vae=["3", 0]),
        "10": node("VAEEncode", pixels=["7", 0], vae=["3", 0]),
        "11": node("KSampler", model=["5", 0], positive=["8", 0],
                   negative=["9", 0], latent_image=["10", 0],
                   seed=seed, steps=STEPS, cfg=CFG,
                   sampler_name="euler", scheduler="simple", denoise=1.0),
        "12": node("VAEDecode", samples=["11", 0], vae=["3", 0]),
        "13": node("SaveImage", images=["12", 0],
                   filename_prefix="drama47_frame/" + shot["name"]),
    }
    return wf


VIEW_MAP = {
    "headshot": "an extreme close-up of the face and shoulders, face filling the frame",
    "hero": "a medium shot from the waist up, subject centered",
    "fullbody": "a full body shot, head to toe fully visible, subject centered",
    "threequarter": "a three-quarter medium shot, subject turned slightly at an angle",
    "profile": "a profile medium shot, subject seen from the side",
    "sad": "a medium close-up, face filling the upper half of the frame",
    "angry": "a medium close-up, face filling the upper half of the frame",
}


def build_prompt(shot):
    cid = shot["char"]
    lock = LOCK[cid]
    scn = SCENE_DESC[shot["scene"]]
    framing = VIEW_MAP.get(shot["view"], "a medium shot")
    return (
        f"Composite the person from Image 1 into the environment of Image 2. "
        f"Keep the exact identity, face, hairstyle and clothing of the person in Image 1 "
        f"perfectly unchanged. The scene, lighting and background must match Image 2. "
        f"The subject is {framing}. "
        f"Subject: {lock}. "
        f"Environment: {scn}. "
        f"Realistic cinematic film still, natural skin texture, sharp focus on the face, "
        f"correct anatomy, natural lighting on the subject matching the scene, "
        f"weathered realistic clothing, film grain. "
        f"Absolutely no text, no letters, no watermark, no logo, no caption anywhere in the image."
    )


def run_one(port, shot, seed):
    name = shot["name"]
    dst = os.path.join(FRAMES, f"{name}.png")
    if os.path.exists(dst) and os.path.getsize(dst) > 20000:
        return (name, True, "cached", 0.0)
    t0 = time.time()
    cid = shot["char"]
    char_path = f"{CAST_ROOT}/{cid}/{cid}_{shot['view']}.png"
    scene_src = SCENES[shot["scene"]]
    scene_port = f"{WORK}/scene_{shot['scene']}.png"
    try:
        portrait_scene(scene_src, scene_port)
        cr = upload_image(port, char_path)
        sr = upload_image(port, scene_port)
        c_ref = cr.get("name") or os.path.basename(char_path)
        s_ref = sr.get("name") or os.path.basename(scene_port)
        prompt = build_prompt(shot)
        wf = build_workflow_lora(c_ref, s_ref, prompt, seed, shot)
        res = post(f"http://127.0.0.1:{port}/prompt", {"prompt": wf})
        pid = res.get("prompt_id")
        if not pid:
            return (name, False, "no prompt_id", time.time() - t0)
        t_wait = time.time()
        last = None
        while time.time() - t_wait < 900:
            time.sleep(3)
            try:
                hist = get_json(f"http://127.0.0.1:{port}/history/{pid}")
            except Exception as e:
                last = f"hist err {e}"
                time.sleep(3)
                continue
            if hist and pid in hist:
                outs = hist[pid].get("outputs", {})
                files = []
                for o in outs.values():
                    files.extend(o.get("images", []))
                if not files:
                    return (name, False, "no images in output", time.time() - t0)
                fs = sorted(files, key=lambda x: x.get("filename", ""))
                src = os.path.join(OUT_ROOT[port], fs[0].get("subfolder", ""), fs[0]["filename"])
                if not os.path.exists(src):
                    src = os.path.join(OUT_ROOT[port], fs[0]["subfolder"], fs[0]["filename"])
                shutil.copy(src, dst)
                # 清理同批次残留，防止旧图冒充新图
                for extra in fs[1:]:
                    try:
                        ex = os.path.join(OUT_ROOT[port], extra.get("subfolder", ""), extra["filename"])
                        if os.path.exists(ex):
                            os.remove(ex)
                    except Exception:
                        pass
                return (name, True, "ok", time.time() - t0)
            last = "waiting"
        return (name, False, f"timeout {last}", time.time() - t0)
    except Exception as e:
        return (name, False, f"err {type(e).__name__}: {e}"[:200], time.time() - t0)


def main():
    only = set(sys.argv[1:])
    shots = [s for s in SHOTS if not only or s["name"] in only]
    print(f"=== 出图 {len(shots)} 张，7 实例并行 ===", flush=True)
    ports = COMFY_PORTS[:]
    results = []
    with ThreadPoolExecutor(max_workers=len(ports)) as ex:
        futs = []
        for i, s in enumerate(shots):
            port = ports[i % len(ports)]
            seed = 20261115 + s["id"] * 101
            futs.append(ex.submit(run_one, port, s, seed))
        for f in futs:
            results.append(f.result())
            print("  ", results[-1], flush=True)
    ok = sum(1 for r in results if r[1])
    print(f"\n完成 {ok}/{len(results)}")
    with open(f"{LOGS}/frames_report.json", "w") as f:
        json.dump([{"name": r[0], "ok": r[1], "msg": r[2], "sec": round(r[3], 1)}
                   for r in results], f, ensure_ascii=False, indent=2)
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
