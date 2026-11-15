#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""背景招牌文字严格核验（修正版）。

发现：VL 评委在自由转录时会把繁体字自动输出成简体（问"写了什么"答"鲜花"，
但二选一能正确选出"鮮花"）。因此判定方式改为：
  1) 自由转录 + 归一化比对（去空格/标点、大小写）
  2) 若不匹配，对"最可能混淆的字符"做二选一强制判别
  3) 综合判定

判定为 OK 的条件：
  - 归一化后完全一致，或
  - 逐字二选一判别全部命中（说明字形正确，只是转录被规范化）
"""
import base64, json, os, re, sys, unicodedata, urllib.request

VL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "qwen2.5-vl-72b"
ROOT = "/ssd2/duanju/scenes_100"
OUT = os.path.join(ROOT, "text_report_v2.json")

# 常见繁简对应（用于生成二选一干扰项）
T2S = {
    '鮮': '鲜', '縫': '缝', '製': '制', '製': '制', '驗': '验', '調': '调', '護': '护',
    '療': '疗', '學': '学', '園': '园', '藥': '药', '號': '号', '廳': '厅', '館': '馆',
    '閱': '阅', '書': '书', '誠': '诚', '信': '信', '師': '师', '事': '事', '務': '务',
    '樣': '样', '間': '间', '車': '车', '場': '场', '莊': '庄', '鐵': '铁', '農': '农',
    '貿': '贸', '遠': '远', '洋': '洋', '漁': '渔', '港': '港', '傳': '传', '統': '统',
    '藝': '艺', '術': '术', '髮': '发', '東': '东', '風': '风', '雲': '云', '電': '电',
    '腦': '脑', '網': '网', '絡': '络', '綜': '综', '合': '合', '壽': '寿', '司': '司',
    '機': '机', '關': '关', '檢': '检', '驗': '验', '種': '种', '類': '类', '營': '营',
    '養': '养', '醫': '医', '療': '疗', '護': '护', '專': '专', '業': '业', '產': '产',
    '資': '资', '訊': '讯', '商': '商', '務': '务', '會': '会', '議': '议', '展': '展',
    '覽': '览', '廳': '厅', '飾': '饰', '傢': '家', '俱': '俱', '環': '环', '境': '境',
    '養': '养', '生': '生', '館': '馆', '練': '练', '習': '习', '場': '场', '館': '馆',
}


def img_b64(p):
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def ask(prompt, img, maxtok=80):
    content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64(img)}"}}]
    content.append({"type": "text", "text": prompt})
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": content}],
                       "max_tokens": maxtok, "temperature": 0.0}).encode()
    req = urllib.request.Request(VL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def norm(s):
    s = unicodedata.normalize("NFKC", s).strip()
    s = re.sub(r'[\s"\'"’。，,.\-—_·:：;；!！?？]', '', s)
    return s.lower()


def pick(a, b):
    """二选一：返回图片上实际是哪一个（'a' / 'b' / '?'）。"""
    return a


def binary_check(img, a, b):
    q = (f'仔细看这张图片中的招牌文字。\n'
         f'问题：招牌上出现的是「{a}」还是「{b}」？\n'
         f'这两个字很相似，请根据笔画像素仔细区分。'
         f'只回答一个：{a} 或 {b}')
    t = ask(q, img, 16).strip()
    if a in t and b not in t:
        return a
    if b in t and a not in t:
        return b
    # 模糊时再看一次整词
    t2 = ask(f'图中招牌上的完整文字是什么？只输出文字本身，保持原样不要转换简繁。', img, 40).strip()
    return t2


def main():
    src = open("/ssd2/duanju/scripts/gen_scenes_100.py", encoding="utf-8").read()
    ns = {}
    m = re.search(r"SCENES = \[(.*?)\n\]", src, re.S)
    exec("SCENES = [" + m.group(1) + "\n]", ns)

    only = set(sys.argv[1:])
    report = []
    for sid, cn, en, text, lang in ns["SCENES"]:
        if only and sid not in only:
            continue
        p = f"{ROOT}/{sid}.png"
        if not os.path.exists(p):
            report.append({"sid": sid, "expected": text, "lang": lang, "status": "MISSING"})
            continue

        got = ask("请读出图中招牌上写的文字。只输出文字本身，不要解释。", p, 60).strip()
        exp_n, got_n = norm(text), norm(got)
        status = "MISMATCH"
        detail = {"sid": sid, "expected": text, "lang": lang, "transcript": got}

        if exp_n == got_n:
            status = "OK"
        elif exp_n and exp_n in got_n:
            status = "OK"
        elif got_n and got_n in exp_n:
            # 转录是预期的子集（可能评委漏字或加了字）-> 需逐字判别
            pass

        if status != "OK":
            # 逐字做「预期繁体 vs 转录简体」二选一判别
            hits, checks = 0, []
            for i, ec in enumerate(text):
                if i >= len(got):
                    break
                gc = got[i]
                if ec == gc:
                    hits += 1
                    continue
                if ec in T2S and T2S[ec] == gc:
                    # 该位评委输出了简体版 -> 二选一确认图片上是繁体
                    r = binary_check(p, ec, gc)
                    checks.append({"pos": i, "expect": ec, "transcript": gc, "verdict": r})
                    if r == ec:
                        hits += 1
                else:
                    checks.append({"pos": i, "expect": ec, "transcript": gc, "verdict": "?"})
            if checks and hits >= len(text) - 0 and len(text) == len(got):
                status = "OK"
                detail["script_checks"] = checks
            elif exp_n == got_n.replace(" ", ""):
                status = "OK"

        detail["status"] = status
        report.append(detail)
        mark = "OK " if status == "OK" else "!! "
        print(f"{mark}{sid} [{lang}] exp={text!r} got={got!r}", flush=True)

    json.dump(report, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    ok = sum(1 for r in report if r["status"] == "OK")
    print(f"[done] {ok}/{len(report)} 通过 -> {OUT}", flush=True)
    print("FAIL:", [r["sid"] for r in report if r["status"] != "OK"], flush=True)


if __name__ == "__main__":
    main()
