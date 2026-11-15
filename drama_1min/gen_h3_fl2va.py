#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H3 FL2VA 对照实验 —— 用已合成底图(frames/*.png)作首帧。

对比 Ref2VA（纯参考图重绘）的差异：
  - 首帧是否精确等于底图（FL2VA 应 100% 保真）
  - 跨镜场景/光线一致性（boardroom 6 镜是否像同一间会议室）
  - 运动量
"""
import os, sys, json, time, uuid, shutil, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shots_1min import SHOTS

BASE = "/ssd2/duanju/drama_1min"
FRAMES = f"{BASE}/frames"
OUT = f"{BASE}/motion_h3_fl8"
OUT_ROOT = {8188: "/ssd2/duanju/comfy_out_8188", 8189: "/ssd2/duanju/comfy_out_8189",
            8190: "/ssd2/duanju/comfy_out_8190", 8191: "/ssd2/duanju/comfy_out_8191",
            8192: "/ssd2/duanju/comfy_out_8192", 8193: "/ssd2/duanju/comfy_out_8193",
            8194: "/ssd2/duanju/comfy_out_8194"}

UNET = "minimax_h3_fl2va_hyperflow_flash_14b-int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VVAE = "minimax_h3_video_vae_fp16.safetensors"
AVAE = "minimax_h3_audio_vae_fp32.safetensors"
W, H, FPS, STEPS, LENGTH = 480, 832, 24, 6, 124

AUDIO_DESC = {
    "s01_hook_fold": "the low hum of a quiet luxury boardroom, faint air conditioning, a pen clicking against paper",
    "s02_ally_lawyer": "quiet boardroom ambience, a single page of paper sliding across polished wood",
    "s03_villain_seat": "boardroom air conditioning hum, the soft creak of a leather chair shifting under weight",
    "s04_exec_silent": "hushed boardroom ambience, a faint wristwatch tick",
    "s05_break_alone": "an empty office at night, distant city traffic far below, a faint air conditioning hum",
    "s06_garage_arrival": "a vast concrete underground garage, echoing footsteps on bare concrete, a distant ventilation fan",
    "s07_rain_street": "heavy rain falling on pavement, water splashing, distant city traffic, a faint car horn far away",
    "s08_villain_rage": "a tense boardroom, a chair scraping back sharply against the floor, restrained room tone",
    "s09_lawyer_cards": "boardroom ambience, the crisp snap of a document folder being opened",
    "s10_exec_reveal": "a quiet boardroom, paper rustling, the faint sound of a chair turning",
    "s11_topfloor_reversal": "a high floor office at night, distant city hum below, faint wind against the window glass",
    "s12_finale_hook": "a high floor office at night, city traffic far below, soft wind across the skyline",
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


def build_prompt(shot):
    action = shot["action"].strip()
    audio = AUDIO_DESC.get(shot["name"], "quiet ambient room tone")
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
        "14": node("SaveVideo", video=["13", 0], filename_prefix="h3fl8/" + tag,
                   format="mp4"),
        "20": node("LoadImage", image=img_ref),
    }


def run_one(port, shot, length):
    name = shot["name"]
    dst = f"{OUT}/{name}.mp4"
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(dst) and os.path.getsize(dst) > 20000:
        return (name, True, "cached", 0.0, port)
    src = f"{FRAMES}/{name}.png"
    t0 = time.time()
    try:
        ur = upload_image(port, src)
        img_ref = ur.get("name") or os.path.basename(src)
        wf = build_workflow(img_ref, build_prompt(shot), 7000 + shot["id"] * 37,
                            name, length)
        r = post(f"http://127.0.0.1:{port}/prompt", {"prompt": wf})
        pid = r["prompt_id"]
        deadline = time.time() + 2400
        got = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/history/{pid}", timeout=30) as h:
                    hd = json.loads(h.read())
                if pid in hd:
                    entry = hd[pid]
                    st = entry.get("status", {})
                    if st.get("status_str") == "error":
                        return (name, False, f"ERROR {str(st.get('messages'))[:300]}",
                                time.time() - t0, port)
                    for o in entry.get("outputs", {}).values():
                        for key in ("images", "videos", "gifs"):
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
            return (name, False, "timeout", time.time() - t0, port)
        shutil.copy(got, dst)
        return (name, True, f"{os.path.getsize(dst)//1024}KB", time.time() - t0, port)
    except Exception as e:
        return (name, False, f"{type(e).__name__}: {e}"[:200], time.time() - t0, port)


def main():
    argv = sys.argv[1:]
    ports = [8192, 8193, 8191, 8190, 8188, 8189, 8194]
    length = LENGTH
    if "--ports" in argv:
        i = argv.index("--ports")
        ports = [int(x) for x in argv[i + 1].split(",") if x.strip().isdigit()] or ports
        argv = argv[:i] + argv[i + 2:]
    if "--length" in argv:
        i = argv.index("--length")
        length = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    targets = argv or [s["name"] for s in SHOTS]
    by_name = {s["name"]: s for s in SHOTS}
    print(f"H3 FL2VA：{len(targets)} 镜 | {length} 帧 @{FPS}fps | 端口 {ports}\n", flush=True)
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=len(ports)) as ex:
        futs = [ex.submit(run_one, ports[i % len(ports)], by_name[n], length)
                for i, n in enumerate(targets)]
        for fu in futs:
            r = fu.result()
            results.append(r)
            flag = "OK " if r[1] else "FAIL"
            print(f"[{flag}] {r[0]:<22} {r[2]:<26} {r[3]:>7.1f}s  port={r[4]}", flush=True)
    total = time.time() - t0
    ok = sum(1 for r in results if r[1])
    print(f"\n完成 {ok}/{len(results)}，总耗时 {total:.1f}s")
    with open(f"{BASE}/h3fl_gen_report.json", "w", encoding="utf-8") as f:
        json.dump([{"name": r[0], "ok": r[1], "info": r[2], "sec": round(r[3], 1)}
                   for r in results], f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
