# -*- coding: utf-8 -*-
"""
《深渊》3分钟短剧 — 多GPU并行渲染
每段: 双参考图(Picture1=人物, Picture2=场景) + 中文台词 -> 视频+中文语音+口型
7 个 ComfyUI 实例 (8188, 8189-8194) 各一个工作线程, 从队列领任务
"""
import json, math, os, re, sys, time, queue, threading
import urllib.request, urllib.parse

sys.path.insert(0, "/ssd2/short_drama")
from storyboard_char16 import REF, SCENES

BASE = "/ssd2/short_drama"
OUT = f"{BASE}/comfy_output/char16_drama"
SEG = f"{OUT}/seg"
os.makedirs(SEG, exist_ok=True)

WF_PATH = "/ssd2/tools/ComfyUI/custom_nodes/ComfyUI-HyperFlow-Flash/example_workflows/api/hyperflow_ref2v_int8_api.json"
PORTS = [8188, 8189, 8190, 8191, 8192, 8193, 8194]
SEG_DUR = 124 / 24.0          # 每段 5.167s
LOG = open(f"{OUT}/render.log", "w", buffering=1)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")


# ---------- 中文台词切分为短句 ----------
def split_line(line, max_chars=11):
    parts = [p for p in re.split(r"([，。！？；…—])", line) if p]
    chunks, cur = [], ""
    for p in parts:
        if len(p) == 1 and p in "，。！？；…—":
            cur += p
        else:
            if cur and len(cur) + len(p) > max_chars:
                chunks.append(cur)
                cur = p
            else:
                cur += p
    if cur:
        chunks.append(cur)
    merged = []
    for c in chunks:
        if not re.search(r"[\u4e00-\u9fff]", c) and merged:
            merged[-1] += c
        else:
            merged.append(c)
    return merged or [line]


# ---------- 构建任务队列: 每镜按 5.167s 一段, 台词均匀铺开 ----------
def build_jobs():
    jobs = queue.Queue()
    for sc in SCENES:
        sid = sc["scene_id"]
        n_segs = sc.get("nsegs") or max(1, math.ceil(sc["dur"] / SEG_DUR))
        pairs = [d.split("：", 1) for d in sc["dialogue"].split("|")]
        chunks = []
        for sp, tx in pairs:
            for ck in split_line(tx):
                chunks.append((sp, ck))
        idx = 0
        for si in range(n_segs):
            take = 0
            if chunks:
                rem_segs = n_segs - si
                rem_chunks = len(chunks) - idx
                if rem_chunks > 0:
                    take = min(rem_chunks, math.ceil(rem_chunks / rem_segs))
            jobs.put({"sid": sid, "seg": si, "n": n_segs, "chunks": chunks[idx:idx + take]})
            idx += take
    return jobs


def http_json(url, data=None, timeout=30):
    if data is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(url, data=json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def download(url, path, timeout=300):
    with urllib.request.urlopen(url, timeout=timeout) as r, open(path, "wb") as f:
        while True:
            b = r.read(1 << 16)
            if not b:
                break
            f.write(b)


def worker(port):
    ep = f"http://127.0.0.1:{port}"
    base = json.load(open(WF_PATH))
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return
        sid, seg, chunks = job["sid"], job["seg"], job["chunks"]
        sc = next(s for s in SCENES if s["scene_id"] == sid)
        out_path = f"{SEG}/seg_s{sid}_{seg}.mp4"
        if os.path.exists(out_path) and os.path.getsize(out_path) > 50000:
            log(f"[P{port}] skip 已存在 s{sid}_{seg}")
            continue
        try:
            wf = json.loads(json.dumps(base))
            wf["20"]["inputs"]["image"] = REF[sc["view"]]
            wf["21"]["inputs"]["image"] = REF[sc["loc"]]
            act = sc["action"]
            if chunks:
                # 一段内可能有多句: 用说话人标签拼接, 只让主要说话人开口
                spk = chunks[0][0]
                text = "".join(t for _, t in chunks)
                act += (f'; {spk} speaks clearly to camera with precise visible lip movement, saying: {text}'
                        f'; cinematic sound design, no subtitles, no on-screen text')
            else:
                act += "; the character remains in scene quietly, ambient sound only, no subtitles, no on-screen text"
            wf["5"]["inputs"]["prompt"] = act
            wf["5"]["inputs"]["length"] = 124
            wf["8"]["inputs"]["noise_seed"] = 8801 + sid * 100 + seg
            wf["14"]["inputs"]["filename_prefix"] = f"hyperflow/char16_drama/seg_s{sid}_{seg}"
            r = http_json(f"{ep}/prompt", {"prompt": wf})
            pid = r["prompt_id"]
            log(f"[P{port}] submit s{sid}_{seg} -> {pid}")
            t0 = time.time()
            done = False
            while time.time() - t0 < 1200:
                time.sleep(5)
                try:
                    h = http_json(f"{ep}/history/{pid}")
                except Exception:
                    continue
                if pid in h:
                    vids = []
                    for _, o in h[pid].get("outputs", {}).items():
                        vids.extend(o.get("videos", []))
                        vids.extend(o.get("images", []))
                    if vids:
                        v = vids[0]
                        q = urllib.parse.urlencode({"filename": v["filename"], "type": "output",
                                                   "subfolder": v.get("subfolder", "")})
                        download(f"{ep}/view?{q}", out_path)
                        log(f"[P{port}] DONE s{sid}_{seg} ({os.path.getsize(out_path)//1024}KB, "
                            f"{int(time.time()-t0)}s)")
                        done = True
                    break
            if not done:
                raise TimeoutError(f"s{sid}_{seg} timeout")
        except Exception as e:
            log(f"[P{port}] FAIL s{sid}_{seg}: {e}")
            jobs.put(job)
            time.sleep(15)


if __name__ == "__main__":
    jobs = build_jobs()
    log(f"共 {jobs.qsize()} 段任务入队, {len(SCENES)} 镜")
    threads = [threading.Thread(target=worker, args=(p,), daemon=True) for p in PORTS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(f"全部完成: 剩余 {jobs.qsize()} 段未完成")
