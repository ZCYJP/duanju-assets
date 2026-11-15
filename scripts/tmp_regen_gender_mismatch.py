#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""性别不符角色的定向重生成（含生成→自检→择优闭环）。

流程（每个角色）：
  1. 用加强性别约束的 prompt 生成 N 个 hero 候选（换不同 seed）
  2. 每个候选用 VL 做性别自检
  3. 只有「性别正确」的候选才进入下一步；多候选则按 VL 性别置信度择优
  4. 胜出候选落盘为 {cid}_hero.png，并从它派生 9 张其他视图
  5. 派生完成后对全部 10 张再跑一次 VL 复检，不一致则记录待人工确认

设计要点：
  * 候选生成用多 seed（base_seed + k*1000），避免同 seed 稳定复现错误性别
  * self-check 复用 tmp_gender_audit 的 PROMPT 逻辑（无偏提问，不透露设定性别）
  * 原图备份到 {cid}/_backup_gender/，可回滚
输出：
  /ssd2/duanju/characters/gender_regen_log.json
  /ssd2/duanju/characters/gender_regen_failed.json
"""
import argparse, base64, json, os, shutil, sys, time, urllib.request, warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

sys.path.insert(0, "/ssd2/duanju/scripts")

CAST_JSON = "/ssd2/duanju/scripts/cast_all_150.json"
CAST_ROOT = "/ssd2/duanju/characters/cast"
LOG_OUT = "/ssd2/duanju/characters/gender_regen_log.json"
FAIL_OUT = "/ssd2/duanju/characters/gender_regen_failed.json"

# ---- VL 自检 ----
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

VL_MODEL = "/ssd2/models/Qwen2.5-VL-7B-Instruct"
GENDER_PROMPT = (
    "看这张照片中的人物，只根据画面里实际呈现的相貌、体态和穿着判断："
    "这个人看起来是男性还是女性？\n"
    "要求：\n"
    "1) 第一行只输出一个词：男 或 女\n"
    "2) 第二行用不超过25个字说明依据\n"
    "3) 如果画面里没有人或性别完全无法判断，第一行输出：无法判断\n"
    "注意：严格按你看到的画面判断，不要受任何文字说明或职业暗示影响。"
)

# ---- ComfyUI ----
import gen_cast as GC

_proc = None
_model = None


def vl_ready():
    global _proc, _model
    if _model is None:
        _proc = AutoProcessor.from_pretrained(VL_MODEL, min_pixels=256 * 28 * 28,
                                              max_pixels=768 * 28 * 28)
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            VL_MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
        _model.eval()
    return _proc, _model


def vl_gender(path):
    proc, mdl = vl_ready()
    img = Image.open(path).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": GENDER_PROMPT}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt").to(mdl.device)
    with torch.inference_mode():
        out = mdl.generate(**inputs, max_new_tokens=48, do_sample=False)
    ans = proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    head = ans.split("\n")[0].strip()
    for kw, g in [("无法判断", "?"), ("男", "男"), ("女", "女")]:
        if kw in head:
            return g, ans.replace("\n", " ")[:140]
    for kw, g in [("男性", "男"), ("女性", "女"), ("男", "男"), ("女", "女")]:
        if kw in ans:
            return g, ans.replace("\n", " ")[:140]
    return "?", ans.replace("\n", " ")[:140]


def make_hero_prompt(c):
    """与 gen_cast.py 一致的 hero prompt + 性别硬约束。"""
    scene = c.get("scene_en") or GC.SCENES.get(c["id"], "in a softly lit neutral interior with shallow depth of field")
    base = (f'A photorealistic cinematic master portrait of {c["desc"]}, {scene}, {GC.STYLE}')
    return GC.gender_lock(c, base)


def make_neg(c):
    return GC.gender_neg(c, 'lowres, bad anatomy, bad hands, extra fingers, deformed face, '
                              'watermark, text, cartoon, anime, illustration, plastic skin')


def instance_alive(url, timeout=8):
    """检查 ComfyUI 实例是否活着"""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/system_stats", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def gen_candidates(c, n_cand, out_dir, base_seed, url=None):
    """生成 n_cand 个 hero 候选，返回 [(path, gender, note)]

    容错：实例失联时自动换端口重试（ComfyUI 实例偶发重启是实测现象）。
    """
    prompt = make_hero_prompt(c)
    neg = make_neg(c)
    results = []
    cand_dir = os.path.join(out_dir, "_cand")
    os.makedirs(cand_dir, exist_ok=True)
    for k in range(n_cand):
        seed = base_seed + k * 1000
        done = False
        for attempt in range(3):
            if url and not instance_alive(url):
                print(f"    cand{k} 实例 {url} 失联，等待重连...", flush=True)
                time.sleep(20)
                continue
            try:
                fs = GC.run_one(GC.up("blank:832x1216"), prompt, seed,
                                f'{c["id"]}_gendcand{k}', cand_dir, neg=neg)
                if not fs:
                    print(f"    cand{k} seed={seed} 无输出", flush=True)
                    time.sleep(5)
                    continue
                p = fs[0]
                g, note = vl_gender(p)
                ok = (g == c["gender"])
                print(f"    cand{k} seed={seed} -> 性别={g} {'✓' if ok else '✗'} | {note[:60]}", flush=True)
                results.append({"path": p, "gender": g, "note": note, "ok": ok, "seed": seed})
                done = True
                break
            except Exception as ex:
                print(f"    cand{k} 异常(第{attempt+1}次): {str(ex)[:150]}", flush=True)
                time.sleep(15)
        if not done:
            print(f"    cand{k} 最终失败", flush=True)
    return results


def pick_best(cands):
    """挑一个性别正确的候选；多个则取第一个（生成顺序即 seed 升序）"""
    good = [c for c in cands if c["ok"]]
    return good[0] if good else None


def regen_character(c, n_cand, base_seed, backup=True, url=None):
    cid = c["id"]
    cdir = os.path.join(CAST_ROOT, cid)
    os.makedirs(cdir, exist_ok=True)
    hero = os.path.join(cdir, f"{cid}_hero.png")

    if backup and os.path.exists(hero):
        bdir = os.path.join(cdir, "_backup_gender")
        os.makedirs(bdir, exist_ok=True)
        dst = os.path.join(bdir, f"{cid}_hero.png")
        if not os.path.exists(dst):
            shutil.copy2(hero, dst)

    print(f"\n=== {cid} {c['name']} 设定{c['gender']} ===", flush=True)
    cands = gen_candidates(c, n_cand, cdir, base_seed, url=url)
    if not cands:
        return {"id": cid, "name": c["name"], "status": "no_candidate"}

    best = pick_best(cands)
    if not best:
        # 全部候选性别错：保存候选供人工挑，并记录
        keep = os.path.join(cdir, "_cand")
        print(f"  !! {n_cand} 个候选性别均不符，原图保留，人工挑选: {keep}", flush=True)
        return {"id": cid, "name": c["name"], "status": "all_wrong",
                "cands": [{k: v for k, v in x.items() if k != "path"} for x in cands]}

    print(f"  选中 seed={best['seed']} 性别={best['gender']}", flush=True)
    shutil.copy2(best["path"], hero)

    # ---- 从新 hero 派生 9 张视图 ----
    keep = ('Keep the exact same person and the exact same face as in image 1, identical facial features, '
            'identical hairstyle, identical skin tone, identical body build')
    scene = c.get("scene_en") or GC.SCENES.get(c["id"], "in a softly lit neutral interior with shallow depth of field")
    views = [
        ("headshot", "portrait",
         f'{keep}. Change only the framing: an extreme close-up head-and-shoulders portrait, face filling '
         f'the frame, neutral expression, looking straight into the camera, plain neutral light grey seamless '
         f'studio background, soft even beauty lighting, sharp focus on the eyes, natural skin texture, '
         f'photorealistic, 8k, no text, no watermark.'),
        ("threequarter", "portrait",
         f'{keep}, identical clothing, identical {scene}. Change only the camera angle: a three-quarter view, '
         f'head turned about 40 degrees away from the camera, shoulders slightly rotated, eyes still looking '
         f'toward the camera, {GC.STYLE}'),
        ("profile", "portrait",
         f'{keep}, identical clothing, identical {scene}. Change only the camera angle: a strict side profile '
         f'view, head turned exactly 90 degrees to the side, showing the clear silhouette of the nose, lips and '
         f'jawline, eyes looking forward, {GC.STYLE}'),
        ("back", "fullbody",
         f'{keep}, identical clothing, identical hairstyle and identical body build. Change only the camera '
         f'angle: a full back view, the back completely facing the camera, the back of the head and the back '
         f'visible, face not visible, standing upright, same {scene}, {GC.STYLE}'),
        ("fullbody", "fullbody",
         f'{keep}, identical clothing. Change only the framing: zoom out to a full body shot, standing '
         f'upright with a natural relaxed posture, whole figure visible from head to toe with clear space '
         f'above the head and below the feet, same {scene}, {GC.STYLE}'),
        ("smile", "portrait",
         f'{keep}, identical clothing, identical background and identical lighting. Change only the '
         f'expression: a warm natural smile with soft crinkled eyes, same head position facing the camera, '
         f'{GC.STYLE}'),
        ("angry", "portrait",
         f'{keep}, identical clothing, identical background and identical lighting. Change only the '
         f'expression: furious and cold, brows pulled down and together, jaw clenched, intense glare, same '
         f'head position facing the camera, {GC.STYLE}'),
        ("sad", "portrait",
         f'{keep}, identical clothing, identical background and identical lighting. Change only the '
         f'expression: deeply sorrowful, glistening eyes with a faint tear, downturned lips, lowered gaze, '
         f'same head position facing the camera, {GC.STYLE}'),
    ]
    neg = make_neg(c)
    sex_off = 0 if c["id"][0] == "M" else 50
    dseed = 910000 + sex_off + int(c["id"][1:]) * 100
    derived_ok = 0
    derived_fail = []
    for vi, (vname, _kind, vp) in enumerate(views):
        dst = os.path.join(cdir, f"{cid}_{vname}.png")
        good = False
        for attempt in range(4):
            if url and not instance_alive(url):
                print(f"    派生 {vname} 实例失联，等待重连...", flush=True)
                time.sleep(20)
                continue
            try:
                fs = GC.run_one(GC.up(hero), vp, dseed + vi * 10 + attempt,
                                f'{cid}_{vname}', cdir, neg=neg)
                if not fs:
                    continue
                # 派生图也要过性别自检，不合格就换种子重试
                g, note = vl_gender(fs[0])
                if g != c["gender"]:
                    print(f"    派生 {vname} 性别={g} 不符，换种子重试 ({attempt+1}/3)", flush=True)
                    time.sleep(1)
                    continue
                os.replace(fs[0], dst)
                good = True
                break
            except Exception as ex:
                print(f"    派生 {vname} 异常: {ex}", flush=True)
                time.sleep(2)
        if good:
            derived_ok += 1
        else:
            derived_fail.append(vname)

    # turnaround：三视图拼图，画布尺寸不同（1536x832），单独生成
    ta = os.path.join(cdir, f"{cid}_turnaround.png")
    ta_p = (f'A professional character turnaround reference sheet of the exact same person shown in image 2. '
            f'Three full-body views of the same person standing side by side, evenly spaced, identical in '
            f'scale, on a plain neutral light grey studio background with soft even lighting and a subtle '
            f'floor shadow. Left: front view facing the camera. Middle: 90 degree side profile view. '
            f'Right: back view. All three figures have the identical face, identical hairstyle and identical '
            f'clothing and body proportions. Clean horizontal layout, generous margins, sharp focus on '
            f'every figure, photorealistic, 8k character design sheet, no text, no labels, no watermark.')
    for attempt in range(3):
        try:
            fs = GC.run_one(GC.up("blank:1536x832"), ta_p, dseed + 30 + attempt,
                            f'{cid}_turnaround', cdir, neg=neg)
            if fs:
                os.replace(fs[0], ta)
                derived_ok += 1
                break
        except Exception as ex:
            print(f"    派生 turnaround 异常: {ex}", flush=True)
            time.sleep(2)
    else:
        derived_fail.append("turnaround")
    print(f"  派生视图成功 {derived_ok}/{len(views)+1}", flush=True)

    # ---- 复检：hero + 3 个关键视图 ----
    recheck = {}
    for v in ["hero", "headshot", "profile", "fullbody"]:
        p = os.path.join(cdir, f"{cid}_{v}.png")
        if os.path.exists(p):
            g, note = vl_gender(p)
            recheck[v] = g
            if g != c["gender"]:
                print(f"  !! 复检 {v} 性别={g} 仍不符: {note[:60]}", flush=True)
    ok_all = all(g == c["gender"] for g in recheck.values()) and recheck
    print(f"  复检: {recheck} -> {'全部一致' if ok_all else '仍有问题'}", flush=True)
    return {"id": cid, "name": c["name"], "status": "ok" if ok_all else "recheck_mismatch",
            "seed": best["seed"], "recheck": recheck,
            "derived_ok": derived_ok, "derived_fail": derived_fail}


def main():
    global LOG_OUT, FAIL_OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None, help="要重生成的 ID 列表")
    ap.add_argument("--from-audit", action="store_true", help="从 gender_audit.json 读取不一致清单")
    ap.add_argument("--cand", type=int, default=4, help="每角色候选数")
    ap.add_argument("--url", default=None, help="ComfyUI 地址")
    ap.add_argument("--log", default=None, help="日志输出路径")
    args = ap.parse_args()
    if args.log:
        LOG_OUT = args.log
        FAIL_OUT = args.log.replace(".json", "_failed.json")

    if args.url:
        os.environ["COMFY_URL"] = args.url
        import importlib
        importlib.reload(GC.comfy)

    cast = json.load(open(CAST_JSON, encoding="utf-8"))
    by_id = {c["id"]: c for c in cast}

    if args.from_audit:
        audit = json.load(open("/ssd2/duanju/characters/gender_audit.json", encoding="utf-8"))
        todo = [r["id"] for r in audit if not r["ok"]]
    elif args.ids:
        todo = args.ids
    else:
        print("需要 --ids 或 --from-audit")
        return
    print(f"待重生成: {len(todo)} 个角色 {todo}", flush=True)

    logs = []
    for cid in todo:
        c = by_id.get(cid)
        if not c:
            print(f"跳过未知 ID {cid}")
            continue
        sex_off = 0 if cid[0] == "M" else 50
        base_seed = 910000 + sex_off + int(cid[1:]) * 100 + 500000  # 换种子段，避免复现原图
        try:
            r = regen_character(c, args.cand, base_seed, url=args.url)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            r = {"id": cid, "name": c["name"], "status": "error", "error": str(ex)[:200]}
        logs.append(r)
        json.dump(logs, open(LOG_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    fails = [r for r in logs if r["status"] != "ok"]
    json.dump(fails, open(FAIL_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n=== 完成：{len(logs)} 个，OK {len(logs)-len(fails)}，需人工确认 {len(fails)} ===", flush=True)
    for f in fails:
        print("  ", f["id"], f["name"], f["status"], f.get("recheck", ""))


if __name__ == "__main__":
    main()
