#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""背景招牌文字核验：逐张用 VL 读取图中招牌文字，与预期字符串比对。

输出 /ssd2/duanju/scenes_100/text_report.json
"""
import base64, json, os, re, sys, urllib.request

VL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen2.5-vl-72b"
ROOT = "/ssd2/duanju/scenes_100"
OUT = os.path.join(ROOT, "text_report.json")


def img_b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def ask(prompt, imgs, maxtok=80):
    content = [{"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64(p)}"}} for p in imgs]
    content.append({"type": "text", "text": prompt})
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}],
                       "max_tokens": maxtok, "temperature": 0.0}).encode()
    req = urllib.request.Request(VL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def norm(s):
    return re.sub(r"\s+", "", s).strip().strip('"\'。.,，')


def main():
    src = open("/ssd2/duanju/scripts/gen_scenes_100.py", encoding="utf-8").read()
    ns = {}
    m = re.search(r"SCENES = \[(.*?)\n\]", src, re.S)
    exec("SCENES = [" + m.group(1) + "\n]", ns)

    only = set(sys.argv[1:])   # 可选：只查指定 sid
    report = []
    for sid, cn, en, text, lang in ns["SCENES"]:
        if only and sid not in only:
            continue
        p = f"{ROOT}/{sid}.png"
        if not os.path.exists(p):
            report.append({"sid": sid, "expected": text, "lang": lang, "status": "MISSING"})
            continue
        got = ask("请读出这张图中招牌上写的文字。只输出文字本身，不要解释、不要加引号。"
                  "如果看不清或没有招牌，输出\"看不清\"。", [p], maxtok=60)
        got = norm(got)
        exp = norm(text)
        if got == exp:
            status = "OK"
        elif exp and exp in got:
            status = "OK"
        elif got in ("看不清", "无", "none", "") or "看不清" in got:
            status = "UNREADABLE"
        else:
            status = "MISMATCH"
        report.append({"sid": sid, "expected": text, "lang": lang, "got": got, "status": status})
        mark = "OK " if status == "OK" else "!! "
        print(f"{mark}{sid} [{lang}] exp={text!r} got={got!r}", flush=True)

    json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ok = sum(1 for r in report if r["status"] == "OK")
    print(f"[done] {ok}/{len(report)} 文字核验通过 -> {OUT}", flush=True)
    print("MISMATCH:", [r["sid"] for r in report if r["status"] == "MISMATCH"], flush=True)


if __name__ == "__main__":
    main()
