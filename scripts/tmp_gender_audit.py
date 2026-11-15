#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""角色形象性别一致性全量检测。

用 Qwen2.5-VL-7B 对每个角色的 hero/headshot/fullbody 三张图做「无偏性别判定」，
再与 cast_all_150.json 的设定性别比对。

关键设计：
  1) 不把设定性别告诉模型（否则模型会顺着答），只问「画面中的人看起来是男是女」。
  2) 只问性别 + 简短依据，max_tokens 小，速度快。
  3) 三个视角多数投票，降低单图误判。
输出：/ssd2/duanju/characters/gender_audit.json
"""
import base64, json, os, re, sys, warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL = "/ssd2/models/Qwen2.5-VL-7B-Instruct"
CAST_ROOT = "/ssd2/duanju/characters/cast"
CAST_JSON = "/ssd2/duanju/scripts/cast_all_150.json"
OUT = "/ssd2/duanju/characters/gender_audit.json"

# 无偏提问：不透露设定性别
PROMPT = (
    "看这张照片中的人物，只根据画面里实际呈现的相貌、体态和穿着判断："
    "这个人看起来是男性还是女性？\n"
    "要求：\n"
    "1) 第一行只输出一个词：男 或 女\n"
    "2) 第二行用不超过25个字说明依据（如：方颌短发男性化/圆润脸型柔和女性化）\n"
    "3) 如果画面里没有人或性别完全无法判断，第一行输出：无法判断\n"
    "注意：严格按你看到的画面判断，不要受任何文字说明或职业暗示影响。"
)

VIEWS = ["hero", "headshot", "fullbody"]

_processor = None
_model = None
_lock = __import__("threading").Lock()


def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def load_model():
    global _processor, _model
    if _model is None:
        with _lock:
            if _model is None:
                _processor = AutoProcessor.from_pretrained(MODEL, min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)
                _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
                _model.eval()
    return _processor, _model


def ask_gender(path):
    try:
        proc, mdl = load_model()
        img = Image.open(path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": PROMPT}],
        }]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], return_tensors="pt").to(mdl.device)
        with torch.inference_mode():
            out = mdl.generate(**inputs, max_new_tokens=48, do_sample=False)
        trimmed = out[:, inputs["input_ids"].shape[1]:]
        ans = proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        return ans
    except Exception as ex:
        return f"[ERROR] {ex}"


def parse_gender(txt):
    if not txt or txt.startswith("[ERROR]"):
        return "?", txt[:120]
    head = txt.split("\n")[0].strip()
    for kw, g in [("无法判断", "?"), ("男", "男"), ("女", "女")]:
        if kw in head:
            return g, txt.replace("\n", " ")[:160]
    # 第一行没命中就全文找
    for kw, g in [("男性", "男"), ("女性", "女"), ("男", "男"), ("女", "女")]:
        if kw in txt:
            return g, txt.replace("\n", " ")[:160]
    return "?", txt.replace("\n", " ")[:160]


def audit_char(c):
    cid = c["id"]
    set_g = c["gender"]
    votes, details = [], {}
    for v in VIEWS:
        p = os.path.join(CAST_ROOT, cid, f"{cid}_{v}.png")
        if not os.path.exists(p):
            details[v] = {"g": "-", "raw": "缺失"}
            continue
        raw = ask_gender(p)
        g, note = parse_gender(raw)
        details[v] = {"g": g, "raw": note}
        if g in ("男", "女"):
            votes.append(g)
    # 多数投票；平票以 hero 为准
    if votes:
        cnt = {"男": votes.count("男"), "女": votes.count("女")}
        if cnt["男"] != cnt["女"]:
            final = max(cnt, key=lambda k: cnt[k])
            vote_txt = f"男{cnt['男']}/女{cnt['女']}"
        else:
            final = details.get("hero", {}).get("g", "?")
            vote_txt = "平票取hero"
    else:
        final, vote_txt = "?", "无有效票"
    ok = (final == set_g)
    return {
        "id": cid, "name": c["name"], "set_gender": set_g, "img_gender": final,
        "ok": ok, "vote": vote_txt, "views": details,
    }


def main():
    cast = json.load(open(CAST_JSON, encoding="utf-8"))
    only = None
    if "--ids" in sys.argv:
        i = sys.argv.index("--ids")
        rest = sys.argv[i+1:]
        only = []
        for a in rest:
            if a.startswith("-"): break
            only.append(a)
    ids = [c["id"] for c in cast]
    cast = [c for c in cast if os.path.isdir(os.path.join(CAST_ROOT, c["id"]))]
    if only:
        cast = [c for c in cast if c["id"] in only]
    print(f"待检测角色: {len(cast)} / {len(ids)}", flush=True)

    results = []
    # 单模型串行（避免显存竞争），但用线程池做图片预处理
    for i, c in enumerate(cast):
        r = audit_char(c)
        results.append(r)
        flag = "OK " if r["ok"] else "!! "
        print(f"[{i+1}/{len(cast)}] {flag}{r['id']} {r['name']} 设定={r['set_gender']} "
              f"图像={r['img_gender']} ({r['vote']})", flush=True)
        if (i + 1) % 10 == 0:
            json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    bad = [r for r in results if not r["ok"]]
    print(f"\n=== 汇总 ===")
    print(f"总数 {len(results)}  一致 {len(results)-len(bad)}  不一致 {len(bad)}", flush=True)
    for r in bad:
        print(f"  {r['id']} {r['name']} 设定{r['set_gender']}→图像{r['img_gender']} ({r['vote']})", flush=True)


if __name__ == "__main__":
    main()
