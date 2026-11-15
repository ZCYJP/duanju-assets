#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《决裂》15秒样本 — 底图合成（Qwen-Image-Edit 2511 + Lightning 4步LoRA）

按短剧语法规范，底图必须精确控制：
  - 景别（人脸高/画面高 目标值）→ 写进 prompt 的构图描述
  - 单人镜 / 双人镜（two_shot）→ 双人镜用 image1=主角色 image2=副角色 image3=场景
  - 过肩镜 → 前景模糊肩部
  - 画面本体 0 文字（OCR 验）

输出：sample_15s/frames/{shot}.png  480x832（归一化后）
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
from shots_15s import SHOTS, LOCK, SCENES, SCENE_DESC, NEGATIVE, CAST_ROOT

BASE = "/ssd2/duanju/drama_1min/sample_15s"
FRAMES = f"{BASE}/frames"
LOGS = f"{BASE}/logs"
os.makedirs(FRAMES, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

COMFY_PORTS = [8191, 8192, 8188, 8189, 8190, 8193, 8194]

UNET = "qwen_image_edit_2511_int8_convrot.safetensors"
CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

W, H = 480, 832
GEN_W, GEN_H = 832, 1248
STEPS = 4
CFG = 1.0
BASE_SEED = 881234


def node(cls, **kw):
    return {"class_type": cls, "inputs": kw}


def upload_image(port, path):
    boundary = "----g15" + uuid.uuid4().hex
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


def post(url, payload, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ---- 景别 → 构图指令（写进 prompt，控制 SCRFD 量到的人脸占比）----
def framing_desc(shot):
    r = shot["face_ratio"]
    fr = shot["framing"]
    if fr == "two_shot":
        if r <= 0.15:
            base = ("medium two-shot, both people visible from the waist up, "
                    "the leading person's face occupying about one seventh of the frame height, "
                    "the second person slightly behind and to the other side, "
                    "both fully inside the frame with clear space around their heads")
        else:
            base = ("medium close two-shot, both people visible from the chest up, "
                    "the leading person's face occupying about one sixth of the frame height, "
                    "the second person clearly visible beside them, "
                    "both fully inside the frame")
        return base
    # 单人镜
    if r >= 0.30:
        return ("extreme close-up, the face fills most of the frame, "
                "the face occupying about one third of the frame height, "
                "cropped just above the chin and at the top of the head, "
                "background completely blurred out")
    if r >= 0.24:
        return ("tight close-up, head and shoulders filling the frame, "
                "the face occupying about a quarter of the frame height, "
                "the top of the head cropped, background heavily blurred")
    if r >= 0.18:
        return ("close-up, head and upper chest visible, "
                "the face occupying about a fifth of the frame height, "
                "background softly blurred")
    return ("medium close-up, chest and shoulders visible with some headroom, "
            "the face occupying about one sixth of the frame height, "
            "the environment readable behind")


def build_prompt(shot):
    cid = shot["char"]
    co = shot.get("co_char")
    fr = shot["framing"]
    scene_desc = SCENE_DESC[shot["scene"]]

    if fr == "two_shot":
        prompt = (
            f"Picture 1 shows a {LOCK[cid]}. Picture 2 shows a {LOCK[co]}. "
            f"Picture 3 shows the environment: {scene_desc}. "
            f"Create a cinematic two-shot photograph in the environment of Picture 3, "
            f"with the character from Picture 1 ({cid}) and the character from Picture 2 ({co}) "
            f"both clearly visible in the same frame, keeping their faces, hair and clothing "
            f"exactly as in the reference pictures. {framing_desc(shot)}. "
            f"The character from Picture 1 is speaking, mouth clearly open mid-word, "
            f"the other character listens silently with a tense expression. "
            f"Both faces fully inside the frame, no face cropped, no face duplicated. "
            f"Single continuous still photograph, shallow depth of field, "
            f"moody night lighting, cold blue window light from behind, warm key light on the faces, "
            f"cinematic colour grade, photorealistic, ultra detailed, 35mm film look. "
            f"Completely blank image with absolutely no text, no letters, no chinese characters, "
            f"no subtitles, no captions, no watermark, no logo, no signature, no stamp, no poster, "
            f"no signage, no signage letters, no UI elements anywhere in the picture."
        )
    else:
        prompt = (
            f"Picture 1 shows a {LOCK[cid]}. Picture 2 shows the environment: {scene_desc}. "
            f"Create a cinematic single-person photograph of the character from Picture 1 "
            f"in the environment of Picture 2, keeping the face, hair and clothing "
            f"exactly as in the reference picture. {framing_desc(shot)}. "
            f"The character is speaking, mouth clearly open mid-word, lips mid-articulation. "
            f"Only one person in the frame. "
            f"Single continuous still photograph, shallow depth of field, "
            f"moody night lighting, cold blue window light from behind, warm key light on the face, "
            f"cinematic colour grade, photorealistic, ultra detailed, 35mm film look. "
            f"Completely blank image with absolutely no text, no letters, no chinese characters, "
            f"no subtitles, no captions, no watermark, no logo, no signature, no stamp, no poster, "
            f"no signage, no signage letters, no UI elements anywhere in the picture."
        )
    return prompt


def build_workflow(refs, prompt, seed, shot):
    """refs: [主角色, 副角色?, 场景] → 2或3张，动态绑定 image1/image2/image3"""
    img_nodes = {"6": "6", "7": "7", "8": "8"}
    loads = {}
    for i, slot in enumerate(["6", "7", "8"]):
        if i < len(refs):
            loads[slot] = node("LoadImage", image=refs[i])
        else:
            loads[slot] = node("LoadImage", image=refs[-1])   # 缺省复用场景

    enc_inputs = {"clip": ["2", 0], "prompt": prompt, "vae": ["3", 0]}
    for i, slot in enumerate(["6", "7", "8"]):
        enc_inputs["image%d" % (i + 1)] = [slot, 0]

    wf = {
        "1": node("UNETLoader", unet_name=UNET, weight_dtype="default"),
        "2": node("CLIPLoader", clip_name=CLIP, type="qwen_image", device="default"),
        "3": node("VAELoader", vae_name=VAE),
        "4": node("LoraLoaderModelOnly", lora_name=LORA, strength_model=1.0, model=["1", 0]),
        "5": node("ModelSamplingAuraFlow", shift=3.1, model=["4", 0]),
        "6": loads["6"],
        "7": loads["7"],
        "8": loads["8"],
        "9": node("TextEncodeQwenImageEditPlus", **enc_inputs),
        "10": node("TextEncodeQwenImageEditPlus", clip=["2", 0], prompt="",
                   image1=["8", 0], vae=["3", 0]),
        "11": node("VAEEncode", pixels=["8", 0], vae=["3", 0]),
        "12": node("KSampler", model=["5", 0], positive=["9", 0], negative=["10", 0],
                   latent_image=["11", 0], seed=seed, steps=STEPS, cfg=CFG,
                   sampler_name="euler", scheduler="simple", denoise=1.0),
        "13": node("VAEDecode", samples=["12", 0], vae=["3", 0]),
        "14": node("SaveImage", images=["13", 0],
                   filename_prefix="drama15s_frame/" + shot["name"]),
    }
    return wf


def find_output(port, prefix, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = post(f"http://127.0.0.1:{port}/history", {"prompt": []})
        except Exception:
            time.sleep(2)
            continue
        time.sleep(2)
        break
    return None


def run_shot(shot, port):
    name = shot["name"]
    out_png = f"{FRAMES}/{name}.png"
    if os.path.exists(out_png):
        print(f"  [skip] {name}", flush=True)
        return {"name": name, "ok": True, "cached": True}
    cid = shot["char"]
    co = shot.get("co_char")
    ref_main = f"{CAST_ROOT}/{cid}/{cid}_{shot['view']}.png"
    scene_ref = SCENES[shot["scene"]]
    refs = [ref_main]
    if co:
        refs.append(f"{CAST_ROOT}/{co}/{co}_headshot.png")
    refs.append(scene_ref)
    refs = refs[:3]

    seed = BASE_SEED + shot["id"] * 7919
    prompt = build_prompt(shot)
    wf = build_workflow(refs, prompt, seed, shot)
    client_id = uuid.uuid4().hex

    try:
        up = []
        for p in refs:
            up.append(upload_image(port, p)["name"])
        for i, slot in enumerate(["6", "7", "8"]):
            wf[slot]["inputs"]["image"] = up[i] if i < len(up) else up[-1]
        r = post(f"http://127.0.0.1:{port}/prompt", {"prompt": wf, "client_id": client_id})
        if "error" in r:
            print(f"  [ERR] {name}: {r['error']}", flush=True)
            return {"name": name, "ok": False, "err": str(r["error"])[:200]}
        pid = r["prompt_id"]
    except Exception as e:
        print(f"  [ERR] {name}: {e}", flush=True)
        return {"name": name, "ok": False, "err": str(e)[:200]}

    # 轮询 history
    t0 = time.time()
    while time.time() - t0 < 400:
        time.sleep(4)
        try:
            h = post(f"http://127.0.0.1:{port}/history/{pid}", {})
        except Exception:
            continue
        if pid in h:
            outs = h[pid].get("outputs", {})
            for k, v in outs.items():
                for im in v.get("images", []):
                    src = im.get("filename", "")
                    sub = im.get("subfolder", "")
                    tt = im.get("type", "output")
                    src_path = f"/ssd2/duanju/comfy_out_{port}/{sub}/{src}" if sub else \
                               f"/ssd2/duanju/comfy_out_{port}/{src}"
                    if os.path.exists(src_path):
                        shutil.copyfile(src_path, out_png)
                        print(f"  [OK] {name} -> {out_png}", flush=True)
                        return {"name": name, "ok": True, "seed": seed, "path": out_png}
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                print(f"  [ERR] {name}: {st}", flush=True)
                return {"name": name, "ok": False, "err": str(st)[:300]}
    print(f"  [TIMEOUT] {name}", flush=True)
    return {"name": name, "ok": False, "err": "timeout"}


def main():
    names = sys.argv[1:]
    shots = [s for s in SHOTS if not names or s["name"] in names]
    ports = [COMFY_PORTS[i % len(COMFY_PORTS)] for i in range(len(shots))]
    print(f"生成 {len(shots)} 张底图，端口 {ports}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(len(shots), 7)) as ex:
        res = list(ex.map(lambda pair: run_shot(pair[0], pair[1]), list(zip(shots, ports))))
    ok = sum(1 for r in res if r["ok"])
    print(f"\n完成 {ok}/{len(shots)}  用时 {time.time()-t0:.1f}s", flush=True)
    json.dump(res, open(f"{LOGS}/frames_report.json", "w"), ensure_ascii=False, indent=2)
    for r in res:
        if not r["ok"]:
            print("  FAILED:", r["name"], r.get("err", "")[:150])


if __name__ == "__main__":
    main()
