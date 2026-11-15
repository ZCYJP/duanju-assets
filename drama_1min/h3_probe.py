#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H3 (MiniMax H3 / HyperFlow) 最小探测 —— 验证 Ref2VA 管线可用性。

测试内容：
  1. ref2va（参考图=人物+场景）能否跑通
  2. 显存是否够、耗时多少
  3. history 里产物怎么回收（SaveVideo 的 outputs 结构）
  4. 原生音频轨是否生成

用法：h3_probe.py <port> [length] [ref_image_size]
"""
import os, sys, json, time, uuid, shutil, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
LENGTH = int(sys.argv[2]) if len(sys.argv) > 2 else 39
REF_SIZE = sys.argv[3] if len(sys.argv) > 3 else "match"

BASE = "/ssd2/duanju/drama_1min"
CHAR_IMG = "/ssd2/duanju/characters/cast/M01/M01_hero.png"
# 场景图已在 ComfyUI input 内，无需上传
SCENE_REL = "char16_drama/scenes/scene_street.png"

OUT_ROOT = {8188: "/ssd2/duanju/comfy_out_8188", 8189: "/ssd2/duanju/comfy_out_8189",
            8190: "/ssd2/duanju/comfy_out_8190", 8191: "/ssd2/duanju/comfy_out_8191",
            8192: "/ssd2/duanju/comfy_out_8192", 8193: "/ssd2/duanju/comfy_out_8193",
            8194: "/ssd2/duanju/comfy_out_8194"}

W, H = 480, 832
STEPS = 6
SEED = 8042

PROMPT = (
    "<Picture 1> is the man in the scene. He walks forward slowly through "
    "<Picture 2>, his coat and hair swaying naturally with the movement. "
    "The camera tracks him from a low angle at shoulder height, dollying "
    "slowly forward. Rain falls in the street, his expression is calm and "
    "determined. Continuous fluid motion throughout, natural human motion, "
    "realistic body dynamics, stable facial features."
)


def node(cls, **kw):
    return {"class_type": cls, "inputs": kw}


def upload_image(port, path):
    boundary = "----h3" + uuid.uuid4().hex
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


def build_workflow(char_ref, prompt, seed, tag):
    return {
        "1": node("HyperFlowLoader",
                  unet_name="minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors"),
        "2": node("CLIPLoader",
                  clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                  type="minimax", device="default"),
        "3": node("VAELoader", vae_name="minimax_h3_video_vae_fp16.safetensors"),
        "4": node("VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors"),
        "5": node("MiniMaxH3ReferenceToVideo", clip=["2", 0], vae=["3", 0],
                  audio_vae=["4", 0], prompt=prompt, width=W, height=H,
                  length=LENGTH, ref_image_size=REF_SIZE,
                  **{
                      "ref_images.ref_image_0": ["20", 0],
                      "ref_images.ref_image_1": ["21", 0],
                  }),
        "6": node("HyperFlowSigmas", model=["1", 0], steps=STEPS),
        "7": node("KSamplerSelect", sampler_name="euler"),
        "8": node("RandomNoise", noise_seed=seed),
        "9": node("BasicGuider", model=["1", 0], conditioning=["5", 0]),
        "10": node("SamplerCustomAdvanced", noise=["8", 0], guider=["9", 0],
                   sampler=["7", 0], sigmas=["6", 0], latent_image=["5", 1]),
        "11": node("VAEDecode", samples=["10", 0], vae=["3", 0]),
        "12": node("VAEDecodeAudio", samples=["10", 0], vae=["4", 0]),
        "13": node("CreateVideo", images=["11", 0], audio=["12", 0], fps=24),
        "14": node("SaveVideo", video=["13", 0], filename_prefix="h3probe/" + tag,
                   format="mp4"),
        "20": node("LoadImage", image=char_ref),
        "21": node("LoadImage", image=SCENE_REL),
    }


def main():
    tag = f"probe_len{LENGTH}_{REF_SIZE}"
    dst = f"{BASE}/ab_out/h3_{tag}.mp4"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    t0 = time.time()
    try:
        up = upload_image(PORT, CHAR_IMG)
        char_ref = up.get("name") or os.path.basename(CHAR_IMG)
        print(f"[{tag}] 角色图上传: {char_ref}", flush=True)
        wf = build_workflow(char_ref, PROMPT, SEED, tag)
        r = post(f"http://127.0.0.1:{PORT}/prompt", {"prompt": wf})
        pid = r["prompt_id"]
        print(f"[{tag}] prompt_id={pid} 提交完成", flush=True)
        deadline = time.time() + 1800
        hd = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{PORT}/history/{pid}", timeout=30) as h:
                    d = json.loads(h.read())
                if pid in d:
                    hd = d[pid]
                    st = hd.get("status", {})
                    if st.get("status_str") == "error":
                        print(f"[{tag}] 错误: {json.dumps(st.get('messages'), ensure_ascii=False)[:1500]}",
                              flush=True)
                        return
                    if hd.get("outputs"):
                        break
            except Exception as e:
                pass
            time.sleep(5)
        if not hd or not hd.get("outputs"):
            print(f"[{tag}] 超时", flush=True)
            return
        print(f"[{tag}] 完成，耗时 {time.time()-t0:.1f}s", flush=True)
        print(f"[{tag}] outputs 结构: {json.dumps(hd['outputs'], ensure_ascii=False)[:1200]}",
              flush=True)
        # 回收产物
        root = OUT_ROOT.get(PORT, "/ssd2/duanju/comfy_out_8188")
        got = None
        for o in hd["outputs"].values():
            for key in ("videos", "gifs", "images"):
                for g in o.get(key, []) or []:
                    fn = g.get("filename", "")
                    if fn.lower().endswith(".mp4"):
                        p = os.path.join(root, g.get("subfolder", ""), fn)
                        if os.path.exists(p) and os.path.getsize(p) > 10000:
                            got = p
                            break
                if got:
                    break
            if got:
                break
        if got:
            shutil.copy(got, dst)
            print(f"[{tag}] 产物: {dst} ({os.path.getsize(dst)/1e6:.2f} MB)", flush=True)
        else:
            print(f"[{tag}] 未找到 mp4，根目录 {root}", flush=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[{tag}] 异常 {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
