#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《决裂》15秒样本 — H3 FL2VA 视频生成

按短剧语法规范：
  - 每镜 length 独立（22/39/56 帧 @24fps = 0.92/1.63/2.33s），实现快切节奏
  - 底图先归一化到 480x832（保证首帧保真 + 构图门禁一致）
  - 原生音轨只作环境层（后续混音会丢掉其人声，只保留环境）
  - seed 每镜唯一（防 ComfyUI 缓存命中跳过写盘）

输出：sample_15s/motion/{shot}.mp4
"""
import os
import sys
import json
import time
import uuid
import shutil
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shots_15s import SHOTS, FPS, W, H

BASE = "/ssd2/duanju/drama_1min/sample_15s"
FRAMES = f"{BASE}/frames"
OUT = f"{BASE}/motion"
os.makedirs(OUT, exist_ok=True)

OUT_ROOT = {8188: "/ssd2/duanju/comfy_out_8188", 8189: "/ssd2/duanju/comfy_out_8189",
            8190: "/ssd2/duanju/comfy_out_8190", 8191: "/ssd2/duanju/comfy_out_8191",
            8192: "/ssd2/duanju/comfy_out_8192", 8193: "/ssd2/duanju/comfy_out_8193",
            8194: "/ssd2/duanju/comfy_out_8194"}

UNET = "minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VVAE = "minimax_h3_video_vae_fp16.safetensors"
AVAE = "minimax_h3_audio_vae_fp32.safetensors"

STEPS = 6
BASE_SEED = 550100

# 场景环境音描述（原生音轨只取环境层，人声由 TTS 提供）
AUDIO_DESC = {
    "boardroom": "a quiet executive boardroom at night, faint air conditioning hum, "
                 "very subtle low room tone, no voices, no music",
}

_upload_lock = threading.Lock()


def node(cls, **kw):
    return {"class_type": cls, "inputs": kw}


def upload_image(port, path):
    boundary = "----h15" + uuid.uuid4().hex
    name = os.path.basename(path)
    data = open(path, "rb").read()
    body = (f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'
            f"Content-Type: image/png\r\n\r\n").encode() + data + \
           f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/upload/image", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with _upload_lock:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())


def post(url, payload, timeout=300):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def build_prompt(shot):
    action = shot["action"].strip()
    audio = AUDIO_DESC.get(shot["scene"], "quiet indoor room tone, no voices, no music")
    return (
        f"{action} "
        f"The character is clearly speaking, lips moving in sync with dialogue, "
        f"with natural blinking and breathing and subtle body micro-movements. "
        f"Continuous fluid motion throughout the whole clip, no frozen frames, "
        f"no static hold, stable facial features with no identity drift. "
        f"Sound: {audio}. "
        f"Realistic cinematic look, natural skin texture, shallow depth of field, "
        f"no text, no subtitles, no watermark, no letters, no logo."
    )


def build_workflow(img_ref, prompt, seed, tag, length):
    return {
        "1": node("HyperFlowLoader", unet_name=UNET),
        "2": node("CLIPLoader", clip_name=CLIP, type="minimax", device="default"),
        "3": node("VAELoader", vae_name=VVAE),
        "4": node("VAELoader", vae_name=AVAE),
        "5": node("MiniMaxH3ImageToVideo", clip=["2", 0], vae=["3", 0],
                  prompt=prompt, width=W, height=H, length=length,
                  first_frame=["20", 0]),
        "6": node("HyperFlowSigmas", model=["1", 0], steps=STEPS),
        "7": node("KSamplerSelect", sampler_name="euler"),
        "8": node("RandomNoise", noise_seed=seed),
        "9": node("BasicGuider", model=["1", 0], conditioning=["5", 0]),
        "10": node("SamplerCustomAdvanced", noise=["8", 0], guider=["9", 0],
                   sampler=["7", 0], sigmas=["6", 0], latent_image=["5", 1]),
        "11": node("VAEDecode", samples=["10", 0], vae=["3", 0]),
        "12": node("VAEDecodeAudio", samples=["10", 0], vae=["4", 0]),
        "13": node("CreateVideo", images=["11", 0], audio=["12", 0], fps=FPS),
        "14": node("SaveVideo", video=["13", 0], filename_prefix="h3_15s/" + tag,
                   format="mp4"),
        "20": node("LoadImage", image=img_ref),
    }


def run_one(port, shot):
    name = shot["name"]
    length = shot["frames"]
    dst = f"{OUT}/{name}.mp4"
    if os.path.exists(dst) and os.path.getsize(dst) > 20000:
        return (name, True, "cached", 0.0, port, length)
    src = f"{FRAMES}/{name}.png"
    if not os.path.exists(src):
        return (name, False, "missing frame", 0.0, port, length)
    t0 = time.time()
    try:
        ur = upload_image(port, src)
        img_ref = ur.get("name") or os.path.basename(src)
        seed = BASE_SEED + shot["id"] * 104729
        wf = build_workflow(img_ref, build_prompt(shot), seed, name, length)
        r = post(f"http://127.0.0.1:{port}/prompt", {"prompt": wf})
        if "error" in r:
            return (name, False, f"ERROR {str(r['error'])[:200]}", time.time()-t0, port, length)
        pid = r["prompt_id"]
        deadline = time.time() + 1800
        got = None
        out_root = OUT_ROOT.get(port, "/ssd2/duanju/comfy_out_8188")
        import glob as _g
        while time.time() < deadline:
            # 方法1：直接扫输出目录（SaveVideo 的 history 结构不稳定，目录最稳）
            for pat in (f"{out_root}/h3_15s/{name}_*.mp4",
                        f"{out_root}/**/{name}_*.mp4"):
                for p in _g.glob(pat, recursive=True):
                    try:
                        if os.path.getsize(p) > 10000 and os.path.getmtime(p) >= t0 - 5:
                            got = p
                            break
                    except OSError:
                        pass
                if got:
                    break
            if got:
                break
            # 方法2：查 history（用于捕捉报错）
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/history/{pid}", timeout=30) as h:
                    hd = json.loads(h.read())
                if pid in hd:
                    entry = hd[pid]
                    st = entry.get("status", {})
                    if st.get("status_str") == "error":
                        return (name, False, f"ERROR {str(st.get('messages'))[:250]}",
                                time.time()-t0, port, length)
                    for o in entry.get("outputs", {}).values():
                        for key in ("videos", "images", "gifs"):
                            for g in o.get(key, []) or []:
                                fn = g.get("filename", "")
                                if fn.lower().endswith(".mp4"):
                                    p = os.path.join(out_root, g.get("subfolder", ""), fn)
                                    if os.path.exists(p) and os.path.getsize(p) > 10000:
                                        got = p
                                        break
                            if got:
                                break
                        if got:
                            break
            except Exception:
                pass
            time.sleep(4)
        if not got:
            return (name, False, "timeout", time.time()-t0, port, length)
        shutil.copy(got, dst)
        return (name, True, f"{os.path.getsize(dst)//1024}KB", time.time()-t0, port, length)
    except Exception as e:
        return (name, False, f"{type(e).__name__}: {e}"[:200], time.time()-t0, port, length)


def main():
    argv = sys.argv[1:]
    ports = [8191, 8192, 8188, 8189, 8190, 8193, 8194]
    if "--ports" in argv:
        i = argv.index("--ports")
        ports = [int(x) for x in argv[i+1].split(",") if x.strip().isdigit()] or ports
        argv = argv[:i] + argv[i+2:]
    targets = argv or [s["name"] for s in SHOTS]
    by_name = {s["name"]: s for s in SHOTS}
    print(f"H3 FL2VA 15s：{len(targets)} 镜 | 各镜独立 length | 端口 {ports}\n", flush=True)
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=len(targets)) as ex:
        futs = [ex.submit(run_one, ports[i % len(ports)], by_name[n])
                for i, n in enumerate(targets)]
        for fu in futs:
            r = fu.result()
            results.append(r)
            flag = "OK " if r[1] else "FAIL"
            print(f"[{flag}] {r[0]:<22} len={r[5]:>3}帧  {r[2]:<24} {r[3]:>7.1f}s  port={r[4]}",
                  flush=True)
    total = time.time() - t0
    ok = sum(1 for r in results if r[1])
    print(f"\n完成 {ok}/{len(results)}，总耗时 {total:.1f}s", flush=True)
    json.dump([{"name": r[0], "ok": r[1], "info": r[2], "sec": round(r[3], 1),
                "frames": r[5], "port": r[4]} for r in results],
              open(f"{BASE}/logs/motion_report.json", "w"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
