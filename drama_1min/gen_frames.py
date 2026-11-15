#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《决裂》1分钟短剧 — 底图合成（角色参考图 + 场景图 → 保脸人物底图）

管线：Qwen-Image-Edit 2511 (int8_convrot) + Lightning 4步LoRA
输入：char16 角色视图PNG（身份锚）+ char16_drama 场景PNG（环境锚）
输出：frames/{shot}.png  832x1248 竖版 人物+场景合成底图

保脸要点：
  - 双图输入：第一张=角色（身份），第二张=场景（环境），prompt 显式声明"图1人物+图2场景"
  - Continuity Lock 英文硬短语原样嵌入
  - Lightning 4步 LoRA + CFG 1.0（与官方 blueprint 一致）
  - 生成后必须 OCR 验 0 伪字（画面不得有文字）

用法：gen_frames.py [shot_name ...]  不带参数 = 全部
"""
import os, sys, json, time, uuid, shutil, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shots_1min import SHOTS, LOCK, SCENES, SCENE_DESC, NEGATIVE, CAST_ROOT

COMFY_PORTS = [8188, 8189, 8190, 8191, 8192, 8193, 8194]
BASE = "/ssd2/duanju/drama_1min"
FRAMES = f"{BASE}/frames"
LOGS = f"{BASE}/logs"
os.makedirs(FRAMES, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

UNET = "qwen_image_edit_2511_int8_convrot.safetensors"
CLIP = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"
LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"

W, H = 832, 1248
STEPS = 4
CFG = 1.0


def node(cls, **kw):
    return {"class_type": cls, "inputs": kw}


def upload_image(port, path):
    boundary = "----gfr" + uuid.uuid4().hex
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


def build_workflow_lora(char_ref, scene_ref, prompt, seed, shot):
    """LoRA 挂在 model 上（KSampler 取 lora 后的 model）"""
    wf = {
        "1": node("UNETLoader", unet_name=UNET, weight_dtype="default"),
        "2": node("CLIPLoader", clip_name=CLIP, type="qwen_image", device="default"),
        "3": node("VAELoader", vae_name=VAE),
        "4": node("LoraLoaderModelOnly", lora_name=LORA, strength_model=1.0,
                  model=["1", 0]),
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
                   filename_prefix="drama1min_frame/" + shot["name"]),
    }
    return wf


def out_root(port):
    return {8188: "/ssd2/duanju/comfy_out_8188", 8189: "/ssd2/duanju/comfy_out_8189",
            8190: "/ssd2/duanju/comfy_out_8190", 8191: "/ssd2/duanju/comfy_out_8191",
            8192: "/ssd2/duanju/comfy_out_8192", 8193: "/ssd2/duanju/comfy_out_8193",
            8194: "/ssd2/duanju/comfy_out_8194"}.get(port, "/ssd2/duanju/comfy_out_8188")


def build_prompt(shot):
    """双图编辑 prompt：图1人物 + 图2场景，合成到目标构图"""
    cid = shot["char"]
    lock = LOCK[cid]
    scn = SCENE_DESC[shot["scene"]]
    view_map = {
        "headshot": "an extreme close-up of the face and shoulders, face filling the frame",
        "hero": "a medium shot from the waist up, subject centered",
        "fullbody": "a full body shot, head to toe fully visible, subject centered",
        "threequarter": "a three-quarter medium shot, subject turned slightly at an angle",
        "sad": "a medium close-up, face filling the upper half of the frame",
        "angry": "a medium close-up, face filling the upper half of the frame",
    }
    framing = view_map.get(shot["view"], "a medium shot")
    return (
        f"Composite the person from Image 1 into the environment of Image 2. "
        f"Keep the exact identity, face, hairstyle and clothing of the person in Image 1 "
        f"perfectly unchanged. The scene, lighting and background must match Image 2. "
        f"The subject is {framing}. "
        f"Subject: {lock}. "
        f"Environment: {scn}. "
        f"Realistic cinematic film still, natural skin texture, sharp focus on the face, "
        f"correct anatomy, natural lighting on the subject matching the scene. "
        f"Absolutely no text, no letters, no watermark, no logo, no caption anywhere in the image."
    )


def run_one(port, shot):
    name = shot["name"]
    dst = os.path.join(FRAMES, f"{name}.png")
    if os.path.exists(dst) and os.path.getsize(dst) > 20000:
        return (name, True, "cached", 0.0)
    t0 = time.time()
    cid = shot["char"]
    char_path = f"{CAST_ROOT}/{cid}/{cid}_{shot['view']}.png"
    scene_path = SCENES[shot["scene"]]
    try:
        cr = upload_image(port, char_path)
        sr = upload_image(port, scene_path)
        c_ref = cr.get("name") or os.path.basename(char_path)
        s_ref = sr.get("name") or os.path.basename(scene_path)
        prompt = build_prompt(shot)
        wf = build_workflow_lora(c_ref, s_ref, prompt, 7000 + shot["id"] * 37, shot)
        r = post(f"http://127.0.0.1:{port}/prompt", {"prompt": wf})
        pid = r["prompt_id"]
        # 等待完成
        deadline = time.time() + 900
        got = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{pid}", timeout=30) as h:
                    hd = json.loads(h.read())
                if pid in hd:
                    st = hd[pid].get("status", {})
                    if st.get("status_str") == "error":
                        return (name, False, f"ERROR {str(st.get('messages'))[:200]}", time.time() - t0)
                    for o in hd[pid].get("outputs", {}).values():
                        for g in o.get("images", []):
                            fn = g.get("filename", "")
                            if fn:
                                p = os.path.join(out_root(port), g.get("subfolder", ""), fn)
                                if os.path.exists(p) and os.path.getsize(p) > 10000:
                                    got = p
                                    break
                        if got:
                            break
                    if got:
                        break
            except Exception:
                pass
            time.sleep(3)
        if not got:
            return (name, False, "timeout", time.time() - t0)
        shutil.copy(got, dst)
        return (name, True, f"{os.path.getsize(dst)//1024}KB", time.time() - t0)
    except Exception as e:
        return (name, False, f"{type(e).__name__}: {e}"[:220], time.time() - t0)


def main():
    targets = sys.argv[1:] or [s["name"] for s in SHOTS]
    by_name = {s["name"]: s for s in SHOTS}
    targets = [t for t in targets if t in by_name]
    print(f"待合成 {len(targets)} 张底图，并行实例 {len(COMFY_PORTS)}", flush=True)
    ok, fail = 0, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(COMFY_PORTS)) as ex:
        futs = {ex.submit(run_one, COMFY_PORTS[i % len(COMFY_PORTS)], by_name[n]): n
                for i, n in enumerate(targets)}
        for f in futs:
            n = futs[f]
            name, good, msg, dt = f.result()
            print(f"  [{'OK ' if good else 'FAIL'}] {name:<22} {msg} {dt:.1f}s", flush=True)
            if good:
                ok += 1
            else:
                fail.append((name, msg))
    print(f"\n完成 {ok}/{len(targets)}，用时 {time.time()-t0:.1f}s", flush=True)
    if fail:
        print("失败清单：")
        for n, m in fail:
            print("  ", n, m)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
