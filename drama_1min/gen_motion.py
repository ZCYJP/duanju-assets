#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《决裂》1分钟短剧 — 静态底图 → Wan2.1 I2V 动态片段

管线（已实测）：Wan2_1-I2V-14B-480P_fp8 + lightx2v 4步 LoRA + dpm++_sde
规格：480x832 竖版 / 80帧 @16fps = 5.0s 每镜 / 7实例并行（8188-8194）
实测：49帧单镜 28.73s，显存峰值 15.8GB

I2V prompt 留白铁律：外观交给首帧，只写运镜/微动态/光影
每镜 prompt 末尾挂 no text / no subtitles / no music

用法：gen_motion.py [shot_name ...]  不带参数 = 全部
     gen_motion.py --ports 8191,8192 s07_rain_street   指定实例（避开被他人占用的端口）
"""
import os, sys, json, time, uuid, shutil, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shots_1min import SHOTS

COMFY_PORTS = [8188, 8189, 8190, 8191, 8192, 8193, 8194]
BASE = "/ssd2/duanju/drama_1min"
FRAMES = f"{BASE}/frames"
OUTDIR = f"{BASE}/motion"
os.makedirs(OUTDIR, exist_ok=True)

MODEL = "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"
LORA = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
VAE = "Wan2_1_VAE_bf16.safetensors"
T5 = "umt5-xxl-enc-bf16.safetensors"

WIDTH, HEIGHT = 480, 832
FPS = 16
STEPS = 4
CFG = 1.0
SHIFT = 5.0

NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，文字，水印，画作，静止不动，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "人物静止不动，画面静止，变形，扭曲，闪烁，色彩突变, "
    "no text, no subtitles, no watermark, no letters, no logo, no caption, no music"
)


def node(cls, **kw):
    return {"class_type": cls, "inputs": kw}


def upload_image(port, path):
    boundary = "----gm" + uuid.uuid4().hex
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


def out_root(port):
    return {8188: "/ssd2/duanju/comfy_out_8188", 8189: "/ssd2/duanju/comfy_out_8189",
            8190: "/ssd2/duanju/comfy_out_8190", 8191: "/ssd2/duanju/comfy_out_8191",
            8192: "/ssd2/duanju/comfy_out_8192", 8193: "/ssd2/duanju/comfy_out_8193",
            8194: "/ssd2/duanju/comfy_out_8194"}.get(port, "/ssd2/duanju/comfy_out_8188")


def build_workflow(img_ref, prompt, seed, shot, frames):
    return {
        "1": node("WanVideoModelLoader", model=MODEL, base_precision="fp16",
                  quantization="fp8_e4m3fn", load_device="offload_device",
                  attention_mode="sdpa"),
        "2": node("LoadWanVideoT5TextEncoder", model_name=T5, precision="bf16",
                  load_device="offload_device", quantization="disabled"),
        "3": node("WanVideoTextEncode", positive_prompt=prompt,
                  negative_prompt=NEG_PROMPT, t5=["2", 0], force_offload=True,
                  use_disk_cache=False, device="gpu"),
        "4": node("WanVideoVAELoader", model_name=VAE, precision="bf16"),
        "5": node("CLIPVisionLoader", clip_name="clip_vision_h.safetensors"),
        "6": node("LoadImage", image=img_ref),
        "7": node("ImageResizeKJv2", image=["6", 0], width=WIDTH, height=HEIGHT,
                  upscale_method="lanczos", keep_proportion="crop",
                  pad_color="0, 0, 0", crop_position="center",
                  divisible_by=16, device="cpu"),
        "8": node("WanVideoClipVisionEncode", clip_vision=["5", 0], image_1=["7", 0],
                  strength_1=1.0, strength_2=1.0, crop="center",
                  combine_embeds="average", force_offload=True),
        "9": node("WanVideoImageToVideoEncode", width=WIDTH, height=HEIGHT,
                  num_frames=frames, noise_aug_strength=0.03,
                  start_latent_strength=1.0, end_latent_strength=1.0,
                  force_offload=True, vae=["4", 0], clip_embeds=["8", 0],
                  start_image=["7", 0]),
        "10": node("WanVideoSampler", model=["1", 0], image_embeds=["9", 0],
                   steps=STEPS, cfg=CFG, shift=SHIFT, seed=seed,
                   force_offload=True, scheduler="dpm++_sde",
                   riflex_freq_index=0, text_embeds=["3", 0]),
        "11": node("WanVideoDecode", vae=["4", 0], samples=["10", 0],
                   enable_vae_tiling=False, tile_x=272, tile_y=272,
                   tile_stride_x=144, tile_stride_y=128),
        "12": node("VHS_VideoCombine", images=["11", 0], frame_rate=FPS,
                   loop_count=0, filename_prefix="drama1min_motion/" + shot,
                   format="video/h264-mp4", pingpong=False, save_output=True),
        "13": node("WanVideoLoraSelect", lora=LORA, strength=1.0),
        "14": node("WanVideoBlockSwap", blocks_to_swap=10, offload_img_emb=True,
                   offload_txt_emb=True, use_non_blocking=False,
                   vace_blocks_to_swap=0, prefetch_blocks=0, block_swap_debug=False),
    }


def run_one(port, shot):
    name = shot["name"]
    dst = os.path.join(OUTDIR, f"{name}.mp4")
    if os.path.exists(dst) and os.path.getsize(dst) > 20000:
        return (name, True, "cached", 0.0)
    t0 = time.time()
    src = os.path.join(FRAMES, f"{name}.png")
    if not os.path.exists(src):
        return (name, False, "缺底图 %s" % src, 0.0)
    try:
        up = upload_image(port, src)
        img_ref = up.get("name") or os.path.basename(src)
        # I2V prompt：动作描述 + 末尾禁文字禁字幕
        prompt = shot["action"] + " no text, no subtitles, no music."
        wf = build_workflow(img_ref, prompt, 8000 + shot["id"] * 53, name, shot["dur"])
        wf["1"]["inputs"]["lora"] = ["13", 0]
        wf["1"]["inputs"]["block_swap_args"] = ["14", 0]
        r = post(f"http://127.0.0.1:{port}/prompt", {"prompt": wf})
        pid = r["prompt_id"]
        deadline = time.time() + 2400
        got = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{pid}", timeout=30) as h:
                    hd = json.loads(h.read())
                if pid in hd:
                    st = hd[pid].get("status", {})
                    if st.get("status_str") == "error":
                        return (name, False, "ERROR " + str(st.get("messages"))[:200], time.time() - t0)
                    for o in hd[pid].get("outputs", {}).values():
                        for key in ("gifs", "videos", "images"):
                            for g in o.get(key, []):
                                fn = g.get("filename", "")
                                if not fn.lower().endswith(".mp4"):
                                    continue
                                p = os.path.join(out_root(port), g.get("subfolder", ""), fn)
                                if os.path.exists(p) and os.path.getsize(p) > 10000:
                                    got = p
                                    break
                            if got:
                                break
                        if got:
                            break
                    if got:
                        break
            except Exception:
                pass
            time.sleep(4)
        if not got:
            return (name, False, "timeout", time.time() - t0)
        shutil.copy(got, dst)
        return (name, True, f"{os.path.getsize(dst)//1024}KB", time.time() - t0)
    except Exception as e:
        return (name, False, f"{type(e).__name__}: {e}"[:220], time.time() - t0)


def main():
    argv = sys.argv[1:]
    ports = list(COMFY_PORTS)
    if "--ports" in argv:
        i = argv.index("--ports")
        raw = argv[i + 1] if i + 1 < len(argv) else ""
        argv = argv[:i] + argv[i + 2:]
        ports = [int(x) for x in raw.split(",") if x.strip().isdigit()]
        if not ports:
            ports = list(COMFY_PORTS)
    targets = argv or [s["name"] for s in SHOTS]
    by_name = {s["name"]: s for s in SHOTS}
    targets = [t for t in targets if t in by_name]
    print(f"待生成 {len(targets)} 段动态，并行实例 {ports}", flush=True)
    ok, fail = 0, []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(ports)) as ex:
        futs = {ex.submit(run_one, ports[i % len(ports)], by_name[n]): n
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
    for n, m in fail:
        print("  FAIL", n, m)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
