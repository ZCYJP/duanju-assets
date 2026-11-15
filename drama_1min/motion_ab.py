#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I2V 运动量 A/B 实验 —— 找出「不像图片配音」的参数组合。

在指定端口上并发跑多组参数，输出每组的：
  - 生成耗时
  - 运动量指标（相邻帧平均差 / 帧间最大差 / 冻结帧数）
  - 首帧保真度（与输入底图的 L1 距离，太大说明跑飞）

用法：motion_ab.py <shot_name> <port> [<port> ...]
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
FRAMES_N = 80

NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，文字，水印，画作，静止不动，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "人物静止不动，画面静止，变形，扭曲，闪烁，色彩突变, "
    "no text, no subtitles, no watermark, no letters, no logo, no caption, no music"
)

# ---- 实验组：每组一个参数组合 ----
EXPERIMENTS = [
    # name, noise_aug, start_latent, end_latent, shift, steps, prompt_suffix
    ("A_baseline", 0.03, 1.0, 1.0, 5.0, 4, ""),
    ("B_shift8", 0.03, 1.0, 1.0, 8.0, 4, ""),
    ("C_motion", 0.10, 0.70, 0.90, 8.0, 4, ""),
    ("D_motion_hi", 0.15, 0.60, 0.85, 8.0, 4, ""),
    ("E_motion_8step", 0.12, 0.70, 0.90, 8.0, 8, ""),
    ("F_motion_camera", 0.12, 0.70, 0.90, 8.0, 4,
     "The camera performs a clear slow dolly-in toward the subject while the "
     "subject makes a visible head turn and shifts body weight, hair and "
     "clothing swaying with the movement, continuous fluid motion throughout."),
]


def node(cls, **kw):
    return {"class_type": cls, "inputs": kw}


def upload_image(port, path):
    boundary = "----ab" + uuid.uuid4().hex
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


def build_workflow(img_ref, prompt, seed, tag, noise, slat, elat, shift, steps):
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
                  num_frames=FRAMES_N, noise_aug_strength=noise,
                  start_latent_strength=slat, end_latent_strength=elat,
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
                   loop_count=0, filename_prefix="ab_motion/" + tag,
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
    name, noise, slat, elat, shift, steps, suffix = exp
    tag = f"{shot['name']}_{name}"
    dst = f"/ssd2/duanju/drama_1min/ab_out/{tag}.mp4"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    src = os.path.join(FRAMES, f"{shot['name']}.png")
    t0 = time.time()
    try:
        up = upload_image(port, src)
        img_ref = up.get("name") or os.path.basename(src)
        prompt = shot["action"] + " no text, no subtitles, no music."
        if suffix:
            prompt = prompt[:-len(" no text, no subtitles, no music.")] + " " + suffix
        wf = build_workflow(img_ref, prompt, 8000 + shot["id"] * 53, tag,
                            noise, slat, elat, shift, steps)
        r = post(f"http://127.0.0.1:{port}/prompt", {"prompt": wf})
        pid = r["prompt_id"]
        deadline = time.time() + 1200
        got = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{pid}", timeout=30) as h:
                    hd = json.loads(h.read())
                if pid in hd:
                    st = hd[pid].get("status", {})
                    if st.get("status_str") == "error":
                        return (name, False, str(st.get("messages"))[:200], 0, {})
                    for o in hd[pid].get("outputs", {}).values():
                        for key in ("gifs", "videos", "images"):
                            for g in o.get(key, []):
                                fn = g.get("filename", "")
                                if fn.lower().endswith(".mp4"):
                                    p = os.path.join(OUT_ROOT.get(port, "/ssd2/duanju/comfy_out_8188"),
                                                     g.get("subfolder", ""), fn)
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
            return (name, False, "timeout", time.time() - t0, {})
        shutil.copy(got, dst)
        return (name, True, dst, time.time() - t0, {})
    except Exception as e:
        return (name, False, f"{type(e).__name__}: {e}"[:200], time.time() - t0, {})


def main():
    shot_name = sys.argv[1]
    ports = [int(x) for x in sys.argv[2:]] or [8191, 8192]
    shot = next(s for s in SHOTS if s["name"] == shot_name)

    print(f"=== I2V 运动量 A/B：{shot_name} ===")
    print(f"实验组 {len(EXPERIMENTS)}，端口 {ports}\n")

    from concurrent.futures import ThreadPoolExecutor
    results = []
    with ThreadPoolExecutor(max_workers=len(ports)) as ex:
        futs = []
        for i, exp in enumerate(EXPERIMENTS):
            futs.append((exp[0], ex.submit(run_one, ports[i % len(ports)], shot, exp)))
        for tag, f in futs:
            nm, ok, msg, dt, _ = f.result()
            print(f"  [{'OK ' if ok else 'FAIL'}] {nm:<16} {dt:>6.1f}s  {msg if not ok else msg}")
            results.append({"exp": nm, "ok": ok, "path": msg if ok else "", "dur": dt})

    print("\n=== 运动量指标 ===")
    print(f"{'实验组':<16} {'耗时':>7} {'相邻帧均差':>10} {'帧间最大差':>10} {'冻结帧':>7} {'首帧保真':>9}")
    metrics = []
    for r in results:
        if not r["ok"]:
            print(f"{r['exp']:<16} {'FAIL':>7}")
            continue
        p = r["path"]
        # 抽全帧
        tmp = tempfile.mkdtemp(prefix="abm_")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", p,
                        os.path.join(tmp, "f_%04d.png")], check=True)
        fs = sorted(os.listdir(tmp))
        fs = [os.path.join(tmp, f) for f in fs]
        try:
            import numpy as np
            from PIL import Image
            arrs = []
            for f in fs:
                arrs.append(np.asarray(Image.open(f).convert("L")).astype(np.float32) / 255.)
            diffs = [float(np.mean(np.abs(arrs[i + 1] - arrs[i]))) for i in range(len(arrs) - 1)]
            frozen = sum(1 for d in diffs if d < 0.002)
            mx = max(diffs) if diffs else 0
            avg = float(np.mean(diffs)) if diffs else 0
            # 首帧保真：与原底图（缩放到 480x832）的 L1
            src = os.path.join(FRAMES, f"{shot['name']}.png")
            if os.path.exists(src):
                im = Image.open(src).convert("L").resize((W, H))
                ref = np.asarray(im).astype(np.float32) / 255.
                fid = float(np.mean(np.abs(arrs[0] - ref)))
            else:
                fid = -1
            metrics.append({"exp": r["exp"], "dur": round(r["dur"], 1),
                            "avg_diff": round(avg, 5), "max_diff": round(mx, 5),
                            "frozen": frozen, "fidelity": round(fid, 4)})
            print(f"{r['exp']:<16} {r['dur']:>6.1f}s {avg:>10.5f} {mx:>10.5f} "
                  f"{frozen:>7} {fid:>9.4f}")
        except Exception as e:
            print(f"{r['exp']:<16} 指标计算失败 {e}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    with open("/ssd2/duanju/drama_1min/ab_out/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("\n指标已写: /ssd2/duanju/drama_1min/ab_out/metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
