#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《零点四十七单》— 动态化（底图 → 视频）

管线：minimax H3 FL2VA（14B int8 + HyperFlow 6步），首帧严格保真 + 原生环境音。
输入：frames_norm/{shot}.png（480x832）
输出：motion/{shot}_s{k}.mp4   124帧@24fps = 5.1667s/段

关键设计：
  - 多段首尾相接（segment chaining）：第 k 段的首帧 = 第 k-1 段的末帧，
    这样一镜可以拉到 15~31 秒而不出现接缝跳变（FL2VA 首帧严格保真）。
  - 7 个 ComfyUI 实例并行，段级并行（每镜的各段同时跑 → 更快）。
  - 历史轮询用 GET（/history/{pid} 只接受 GET，POST 会 405）。

用法：
  gen_motion.py                      # 全部镜，每镜 3 段
  gen_motion.py --segments 1         # 每镜 1 段（快速验证）
  gen_motion.py s01_a01_night_ride   # 指定镜位
"""
import os
import sys
import json
import time
import uuid
import shutil
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from story import SHOTS

BASE = "/ssd2/duanju/drama_midnight47"
FRAMES = f"{BASE}/frames_norm"
OUT = f"{BASE}/motion"
SEGS = f"{BASE}/motion_segs"
OUT_ROOT = {8188: "/ssd2/duanju/comfy_out_8188", 8189: "/ssd2/duanju/comfy_out_8189",
            8190: "/ssd2/duanju/comfy_out_8190", 8191: "/ssd2/duanju/comfy_out_8191",
            8192: "/ssd2/duanju/comfy_out_8192", 8193: "/ssd2/duanju/comfy_out_8193",
            8194: "/ssd2/duanju/comfy_out_8194"}

UNET = "minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VVAE = "minimax_h3_video_vae_fp16.safetensors"
AVAE = "minimax_h3_audio_vae_fp32.safetensors"
W, H, FPS, STEPS, LENGTH = 480, 832, 24, 6, 124

# 环境音描述（喂给 FL2VA 生成原生音轨）
AUDIO_DESC = {
    "s01_a01_night_ride":
        "heavy rain falling on pavement, water splashing under wheels, distant city traffic, "
        "the hum of an electric scooter motor, faint rain on a helmet visor",
    "s01_a02_phone_screen":
        "heavy rain hammering a metal awning, water dripping off a visor, distant muffled traffic, "
        "a very faint electronic notification chime",
    "s01_a03_demolished":
        "steady water dripping from a broken concrete ceiling, hollow wind through exposed rebar, "
        "distant muffled city noise, a faint metallic creak",
    "s01_a04_final_call":
        "rain falling steadily in a street lamp glow, close droplets hitting a helmet, "
        "a distant car engine fading away, muffled traffic",
    "s02_b01_figure_under_umbrella":
        "heavy rain on a black umbrella canopy, water running off the rim, hollow dripping in an "
        "empty demolished concrete building, distant wind through bare columns",
    "s02_b02_turn_and_face":
        "rain falling behind her, light tapping on an umbrella canopy, quiet hollow room tone of an "
        "empty concrete shell, distant city rumble",
    "s02_b03_give_notebook":
        "rain dripping from an umbrella rim, a faint creak of old paper being handled, quiet hollow "
        "room tone, distant city noise",
    "s02_b04_rider_reacts":
        "rain on a yellow helmet and jacket, close droplets hitting fabric, hollow dripping in an "
        "empty concrete building, distant muted traffic",
    "s02_b05_notebook_page":
        "close rain on a wet jacket, quiet page rustling, hollow dripping in an empty concrete shell, "
        "very distant city rumble",
}


def node(cls, **kw):
    return {"class_type": cls, "inputs": kw}


def upload_image(port, path):
    boundary = "----h3f" + uuid.uuid4().hex
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
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def build_prompt(shot, seg_idx):
    action = shot["action"].strip()
    audio = AUDIO_DESC.get(shot["name"], "quiet ambient room tone")
    # 第 2 段起强调「延续上一段的运动与状态」，避免接缝处人物状态突变
    cont = ""
    if seg_idx > 0:
        cont = (" The motion continues naturally from the previous moment, "
                "the subject stays in the same position and emotional state. ")
    return (
        f"{action} "
        f"{cont}"
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
        "14": node("SaveVideo", video=["13", 0], filename_prefix="drama47_motion/" + tag,
                   format="mp4"),
        "20": node("LoadImage", image=img_ref),
    }


def run_one(port, shot, seg_idx, first_frame_path, seed):
    """生成一段视频，返回 (name, ok, msg, sec, out_path)"""
    tag = f"{shot['name']}_s{seg_idx}"
    dst = f"{SEGS}/{tag}.mp4"
    os.makedirs(SEGS, exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 20000:
        return (tag, True, "cached", 0.0, dst)
    t0 = time.time()
    try:
        ur = upload_image(port, first_frame_path)
        img_ref = ur.get("name") or os.path.basename(first_frame_path)
        wf = build_workflow(img_ref, build_prompt(shot, seg_idx), seed, tag, LENGTH)
        r = post(f"http://127.0.0.1:{port}/prompt", {"prompt": wf})
        pid = r["prompt_id"]
        deadline = time.time() + 2400
        got = None
        while time.time() < deadline:
            try:
                hd = get_json(f"http://127.0.0.1:{port}/history/{pid}", timeout=30)
                if pid in hd:
                    entry = hd[pid]
                    st = entry.get("status", {})
                    if st.get("status_str") == "error":
                        return (tag, False, f"ERROR {str(st.get('messages'))[:250]}",
                                time.time() - t0, None)
                    for o in entry.get("outputs", {}).values():
                        for key in ("videos", "images", "gifs"):
                            for g in o.get(key, []) or []:
                                fn = g.get("filename", "")
                                if fn.lower().endswith(".mp4"):
                                    p = os.path.join(
                                        OUT_ROOT.get(port, "/ssd2/duanju/comfy_out_8188"),
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
            return (tag, False, "timeout", time.time() - t0, None)
        shutil.copy(got, dst)
        return (tag, True, f"{os.path.getsize(dst)//1024}KB", time.time() - t0, dst)
    except Exception as e:
        return (tag, False, f"{type(e).__name__}: {e}"[:200], time.time() - t0, None)


def extract_last_frame(mp4_path, dst_png):
    """抽取视频最后一帧，作为下一段的首帧"""
    import subprocess
    cmd = ["/ssd2/miniconda3/envs/videogen/bin/ffmpeg", "-y", "-sseof", "-0.2",
           "-i", mp4_path, "-frames:v", "1", "-q:v", "2", dst_png]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dst_png):
        return None
    return dst_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shots", nargs="*", help="指定镜位名，空 = 全部")
    ap.add_argument("--segments", type=int, default=3, help="每镜分段数")
    ap.add_argument("--ports", default="8188,8189,8190,8191,8192,8193,8194")
    args = ap.parse_args()
    ports = [int(x) for x in args.ports.split(",") if x.strip().isdigit()]
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SEGS, exist_ok=True)

    targets = args.shots or [s["name"] for s in SHOTS]
    by_name = {s["name"]: s for s in SHOTS}
    print(f"H3 FL2VA：{len(targets)} 镜 × {args.segments} 段 | "
          f"{LENGTH}帧@{FPS}fps | 端口 {ports}\n", flush=True)

    results = []
    # 逐镜处理：镜内分段必须串行（后段首帧依赖前段末帧），镜间并行
    def do_shot(shot):
        shot_res = []
        ff = f"{FRAMES}/{shot['name']}.png"
        for k in range(args.segments):
            if not os.path.exists(ff):
                shot_res.append((f"{shot['name']}_s{k}", False, "缺底图", 0.0, None))
                break
            seed = 20261115 + shot["id"] * 101 + k * 17
            port = ports[(shot["id"] * 3 + k) % len(ports)]
            tag, ok, msg, sec, path = run_one(port, shot, k, ff, seed)
            shot_res.append((tag, ok, msg, sec, path))
            if not ok:
                break
            if k + 1 < args.segments:
                nxt = f"{SEGS}/{shot['name']}_s{k+1}_ff.png"
                got = extract_last_frame(path, nxt)
                if got:
                    ff = got
                else:
                    print(f"  [warn] {shot['name']} 末帧抽取失败，下段仍用原底图", flush=True)
        return shot_res

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=min(len(ports), len(targets))) as ex:
        futs = {ex.submit(do_shot, by_name[n]): n for n in targets}
        for fu in futs:
            for r in fu.result():
                results.append(r)
                flag = "OK " if r[1] else "FAIL"
                print(f"[{flag}] {r[0]:<32} {r[2]:<26} {r[3]:>7.1f}s", flush=True)

    total = time.time() - t0
    ok = sum(1 for r in results if r[1])
    print(f"\n完成 {ok}/{len(results)}，总耗时 {total:.1f}s")
    with open(f"{BASE}/logs/motion_report.json", "w", encoding="utf-8") as f:
        json.dump([{"name": r[0], "ok": r[1], "info": r[2], "sec": round(r[3], 1)}
                   for r in results], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
