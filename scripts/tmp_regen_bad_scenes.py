#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重生成 9 张含伪文字的背景图。

背景：四层审计（OCR + VL全图 + VL招牌 + 放大复核）判定 scenes_100/ 下 9 张背景含 AI
生成的乱码字符，不满足"背景中的汉字一定要是真实汉字"的要求。

做法：
1. 从 scenes_100_text_final.json 读出 verdict == 伪文字 的 sid 列表
2. 复用 gen_scenes_100.py 的 run_one/up/NEG/STYLE 与提示词结构，但强化文字约束
   （强调只允许出现指定招牌文字、其余位置保持无字）
3. 按 shard/of 分片到多台 ComfyUI 实例并行重生成
4. 输出到 scenes_100_regen/，不覆盖原图；核验通过后再人工决定是否替换

用法: python3 tmp_regen_bad_scenes.py --shard 0 --of 7 --url http://127.0.0.1:8188 --out /ssd2/duanju/scenes_100_regen
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gen_scenes_100 as G

AUDIT = "/ssd2/duanju/scenes_100_text_final.json"


def load_bad():
    data = json.load(open(AUDIT, encoding="utf-8"))
    bad = []
    for it in data["items"]:
        if it.get("verdict") == "伪文字":
            bad.append(it["sid"])
    # 只重生成确实在 scenes_100/ 里存在、且当前内容有问题的
    bad = [s for s in bad if os.path.exists(f"/ssd2/duanju/scenes_100/{s}.png")]
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--url", default=None)
    ap.add_argument("--out", default="/ssd2/duanju/scenes_100_regen")
    args = ap.parse_args()

    if args.url:
        os.environ["COMFY_URL"] = args.url
        import importlib
        importlib.reload(G.comfy)

    os.makedirs(args.out, exist_ok=True)
    log = open(f"{args.out}/regen_shard{args.shard}.log", "a", encoding="utf-8")

    def say(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    bad = load_bad()
    scenes = {s[0]: s for s in G.SCENES}
    mine = [s for i, s in enumerate(bad) if i % args.of == args.shard]

    say(f"=== shard {args.shard}/{args.of}: {len(mine)}/{len(bad)} bad scenes on {G.comfy.URL} ===")
    say(f"    targets: {mine}")

    for sid in mine:
        _, cn, en, text, lang = scenes[sid]
        dst = f"{args.out}/{sid}.png"
        if os.path.exists(dst) and os.path.getsize(dst) > 20000:
            say(f"  {sid} exists, skip")
            continue

        # 强化约束：整幅画面只允许出现这一处招牌文字，且必须逐字符正确；
        # 明确禁止任何额外小字、数字串、乱码字符。
        prompt = (
            f"A photorealistic cinematic background of {en}. "
            f"The scene is empty of people. "
            f"There is exactly ONE sign in the whole scene, placed prominently, "
            f"and it displays exactly this text: \"{text}\". "
            f"The text is written in {lang}. "
            f"Every single character of the sign is correct and legible, "
            f"spelled exactly as given, with sharp clean strokes on a realistic signboard. "
            f"Absolutely no other text anywhere in the image: "
            f"no small print, no license plate numbers, no poster subtitles, "
            f"no random letters, no meaningless glyphs, no garbled characters. "
            f"Clean surfaces only. {G.STYLE}"
        )
        seed = 990000 + abs(hash(sid)) % 90000
        ok = False
        for attempt in range(4):
            try:
                fs = G.run_one(G.up("blank:1216x832"), prompt,
                               seed + attempt * 13, f"regen_{sid}", args.out, G.NEG)
                if fs:
                    os.replace(fs[0], dst)
                    ok = True
                    say(f"  {sid} -> {dst} [{lang}: {text}] attempt={attempt}")
                    break
            except Exception as ex:
                say(f"  {sid} attempt {attempt} failed: {ex}")
                time.sleep(3)
        if not ok:
            say(f"  {sid} FAILED")

    say("=== shard finished ===")


if __name__ == "__main__":
    main()
