#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
帧数/步数 → 耗时 标定实验。
目的：确定「更短更多的镜头」是否比「更长的镜头」更快（固定总时长下）。
"""
import os, sys, json, time, uuid, shutil, subprocess, tempfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shots_1min import SHOTS

BASE = "/ssd2/duanju/drama_1min"
FRAMES = f"{BASE}/frames"
OUT_ROOT = {8188: "/ssd2/duanju/comfy_out_8188", 8189: "/ssd2/duanju/comfy_out_8189",
            8190: "/ssd2/duanju/comfy_out_8190", 8191: "/ssd2/duanju/comfy_out_8191",
            8192: "/ssd2/duanju/comfy_out_8192", 8193: "/ssd2/duanju/comfy_out_8193",
            8194: "/ssd2/duanju/comfy_out_8194"}
MODEL = "Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors"
LORA = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
VAE = "Wan2_1_VAE_bf16.safetensors"
T5 = "umt5-xxl-enc-bf16.safetensors"
W, H = 480, 832
FPS = 16
NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，文字，水印，画作，静止不动，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "人物静止不动，画面静止，变形，扭曲，闪烁，色彩突变, "
    "no text, no subtitles, no watermark, no letters, no logo, no caption, no music"
)

# (tag, num_frames, steps, noise, slat, shift)
EXPERIMENTS = [
    ("G48_4step", 48, 4, 0.03, 1.0, 5.0),
    ("H48_8step", 48, 8, 0.03, 1.0, 5.0),
    ("I64_8step", 64, 8, 0.03, 1.0, 5.0),
    ("J48_8step_s8", 48, 8, 0.03, 1.0, 8.0),
]


def node(cls, **kw):
    return {"class_type": cls, "inputs": kw}


def upload_image(port, path):
    boundary = "----fa" + uuid.uuid4().hex
    data = open(path, "rb").read()
    body = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{os.path.basename(path)}"\r\n'
            f"Content-Type: image/png\r\n\r\n").encode() + data + \
           f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def build_workflow(img_ref, prompt, seed, tag, nf, steps, noise, slat, shift):
    wf = {
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
        "7": node("ImageResizeKJv2", image=["6", 0], width=W, height=H,
                  upscale_method="lanczos", keep_proportion="crop",
                  pad_color="0, 0, 0", crop_position="center",
                  divisible_by=16, device="cpu"),
        "8": node("WanVideoClipVisionEncode", clip_vision=["5", 0], image_1=["7", 0],
                  strength_1=1.0, strength_2=1.0, crop="center",
                  combine_embeds="average", force_offload=True),
        "9": node("WanVideoImageToVideoEncode", width=W, height=H,
                  num_frames=nf, noise_aug_strength=noise,
                  start_latent_strength=slat, end_latent_strength=1.0,
                  force_offload=True, vae=["4", 0], clip_embeds=["8", 0],
                  start_image=["7", 0]),
        "10": node("WanVideoSampler", model=["1", 0], image_embeds=["9", 0],
                   steps=steps, cfg=1.0, shift=shift, seed=seed,
                   force_offload=True, scheduler="dpm++_sde",
                   riflex_freq_index=0, text_embeds=["3", 0]),
        "11": node("WanVideoDecode", vae=["4", 0], samples=["10", 0],
                   enable_vae_tiling=False, tile_x=272, tile_y=272,
                   tile_stride_x=144, tile_stride_y=128),
        "12": node("VHS_VideoCombine", images=["11", 0], frame_rate=FPS,
                   loop_count=0, filename_prefix="ab_frames/" + tag,
                   format="video/h264-mp4", pingpong=False, save_output=True),
        "13": node("WanVideoLoraSelect", lora=LORA, strength=1.0),
        "14": node("WanVideoBlockSwap", blocks_to_swap=10, offload_img_emb=True,
                   offload_txt_emb=True, use_non_blocking=False,
                   vace_blocks_to_swap=0, prefetch_blocks=0, block_swap_debug=False),
    }
    wf["1"]["inputs"]["lora"] = ["13", 0]
    wf["1"]["inputs"]["block_swap_args"] = ["14", 0]
    return wf


def run_one(port, shot, exp):
    tag, nf, steps, noise, slat, shift = exp
    dst = f"/ssd2/duanju/drama_1min/ab_out/{tag}.mp4"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    src = os.path.join(FRAMES, f"{shot['name']}.png")
    t0 = time.time()
    try:
        up = upload_image(port, src)
        img_ref = up.get("name") or os.path.basename(src)
        prompt = shot["action"] + " no text, no subtitles, no music."
        wf = build_workflow(img_ref, prompt, 8000 + shot["id"] * 53, tag,
                            nf, steps, noise, slat, shift)
        r = post(f"http://127.0.0.1:{port}/prompt", {"prompt": wf})
        pid = r["prompt_id"]
        deadline = time.time() + 900
        got = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{pid}", timeout=30) as h:
                    hd = json.loads(h.read())
                if pid in hd:
                    st = hd[pid].get("status", {})
                    if st.get("status_str") == "error":
                        return (tag, False, str(st.get("messages"))[:150], 0)
                    for o in hd[pid].get("outputs", {}).values():
                        for key in ("gifs", "videos", "images"):
                            for g in o.get(key, []):
                                if g.get("filename", "").lower().endswith(".mp4"):
                                    p = os.path.join(
                                        OUT_ROOT.get(port, "/ssd2/duanju/comfy_out_8188"),
                                        g.get("subfolder", ""), g["filename"])
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
            time.sleep(3)
        if not got:
            return (tag, False, "timeout", time.time() - t0)
        shutil.copy(got, dst)
        return (tag, True, dst, time.time() - t0)
    except Exception as e:
        return (tag, False, f"{type(e).__name__}: {e}"[:150], time.time() - t0)


def main():
    shot_name = sys.argv[1]
    ports = [int(x) for x in sys.argv[2:]] or [8191, 8192, 8188]
    shot = next(s for s in SHOTS if s["name"] == shot_name)
    from concurrent.futures import ThreadPoolExecutor
    print(f"=== 帧数/步数标定：{shot_name} ===\n")
    results = []
    with ThreadPoolExecutor(max_workers=len(ports)) as ex:
        futs = [(e[0], ex.submit(run_one, ports[i % len(ports)], shot, e))
                for i, e in enumerate(EXPERIMENTS)]
        for tag, f in futs:
            nm, ok, msg, dt = f.result()
            print(f"  [{'OK ' if ok else 'FAIL'}] {nm:<16} {dt:>6.1f}s")
            results.append({"exp": nm, "ok": ok, "path": msg if ok else "", "dur": dt})

    print(f"\n{'实验组':<16} {'帧数':>5} {'步数':>5} {'耗时':>8} {'s/帧':>8} {'视频时长':>9} {'运动均差':>9}")
    for r, e in zip(results, EXPERIMENTS):
        if not r["ok"]:
            print(f"{r['exp']:<16} FAIL {e[1]}帧 {e[2]}步")
            continue
        tmp = tempfile.mkdtemp(prefix="fam_")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", r["path"],
                        os.path.join(tmp, "f_%04d.png")], check=True)
        fs = sorted(os.listdir(tmp))
        try:
            import numpy as np
            from PIL import Image
            arrs = [np.asarray(Image.open(os.path.join(tmp, f)).convert("L")).astype(np.float32) / 255.
                    for f in fs]
            diffs = [float(np.mean(np.abs(arrs[i + 1] - arrs[i]))) for i in range(len(arrs) - 1)]
            avg = float(np.mean(diffs)) if diffs else 0
            print(f"{r['exp']:<16} {e[1]:>5} {e[2]:>5} {r['dur']:>7.1f}s "
                  f"{r['dur']/e[1]:>8.3f} {len(arrs)/FPS:>8.2f}s {avg:>9.5f}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
