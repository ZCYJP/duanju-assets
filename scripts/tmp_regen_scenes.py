#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""背景文字重生成：针对文字核验不通过的场景，强化文字约束后重跑。

文字问题的三类根因与对策：
  1) 繁简被"自动规范化"成简体 -> prompt 明确写 "Traditional Chinese characters only,
     do NOT simplify any character"，并把繁体写法直接写进引号紧邻说明
  2) 繁体/汉字错字 -> 逐字列出期望字形："The sign must contain exactly these characters
     in this order: 誠 信 律 師 事 務 所"
  3) 日/韩/英文字形崩坏 -> 用简短、笔画少的词替换，并对这类语言改用
     "photographic close-up of the sign" 强化

用法：python3 tmp_regen_scenes.py --sids a,b,c [--url ...] [--of N --shard I]
"""
import argparse, json, os, re, sys, time

sys.path.insert(0, '/ssd2/short_drama/dh_scripts')
sys.path.insert(0, '/ssd2/duanju/scripts')
import comfy
import importlib
GS = importlib.import_module('gen_scenes_100')

ROOT = "/ssd2/duanju/scenes_100"

# 对难字语言做替换：改成更短、更常见、字形更稳的表达
#
# 核心教训（来自 tmp_probe_script.py 的诊断）：
#   模型常把繁体字写成"日语汉字形"（樣->様、驗->験、險->険），
#   日文假名也会音近错字（ブティック->ブテック、ペット->バート）。
#   而简体汉字 / 短英文单词的字形歧义最小，出错率最低。
# 因此这批失败场景一律改用"字形零歧义"写法：
#   繁体 -> 简体（等价语义，字形无中日歧义）
#   韩文 -> 简体（谚文音节块对扩散模型极不友好）
#   日文 -> 短英文或简体
#
# 注意：语言名必须与 build_prompt 里的判定字符串完全一致
#   （'英语' / '日语' / '韩语' / '法语'），否则会掉进兜底分支、
#   丢掉全部文字强化约束 —— 这是上一轮的真 bug。
REPLACE = {
    'bg_delivery_street': ('外賣専門', '外卖驿站', '简体中文'),
    'bg_yoga_studio': ('요가 스튜디오', '瑜伽馆', '简体中文'),
    'bg_airport': ('출발大厅', '出发大厅', '简体中文'),
    'bg_cafe': ('珈琲のひととき', 'COFFEE', '英语'),
    'bg_support_centre': ('고객센터', '客服中心', '简体中文'),
    'bg_boarding_gate': ('게이트', '登机口', '简体中文'),
    'bg_insurance': ('保険会社', '保险公司', '简体中文'),
    'bg_boutique': ('ブティック', '精品店', '简体中文'),
    'bg_sample_room': ('樣衣間', '样衣间', '简体中文'),
}


def build_prompt(en, text, lang):
    """按语言类型定制文字约束强度。"""
    is_cjk = lang in ('简体中文', '繁体中文', '日语', '韩语', '法语')
    if lang == '繁体中文':
        chars = ' '.join(text)
        return (f'A photorealistic cinematic background of {en}. '
                f'The scene is empty of people. '
                f'A large simple rectangular signboard with big bold characters fills a clear area of the scene. '
                f'The sign displays Traditional Chinese characters, written EXACTLY as: "{text}". '
                f'Each character is clearly separated and must appear in this exact order: {chars}. '
                f'Use Traditional Chinese glyphs only — do NOT convert them to Simplified Chinese, '
                f'do NOT substitute any character with a similar-looking one. '
                f'Correct, legible, sharp strokes on a realistic signboard. '
                f'{GS.STYLE}')
    if lang == '简体中文':
        chars = ' '.join(text)
        return (f'A photorealistic cinematic background of {en}. '
                f'The scene is empty of people. '
                f'A large simple rectangular signboard with big bold characters fills a clear area of the scene. '
                f'The sign displays Simplified Chinese characters, written EXACTLY as: "{text}". '
                f'Each character is clearly separated and must appear in this exact order: {chars}. '
                f'Correct, legible, sharp strokes on a realistic signboard. '
                f'{GS.STYLE}')
    if lang == '英语':
        chars = ' '.join(text)
        return (f'A photorealistic cinematic background of {en}. '
                f'The scene is empty of people. '
                f'A large simple rectangular signboard with big bold characters fills a clear area of the scene. '
                f'The sign displays ONLY this single English word: "{text}". '
                f'Write it in capital letters exactly as: {chars}. '
                f'One single short word, no additional letters, no extra text, '
                f'spelled correctly letter by letter, no misspelled letters, no random symbols, '
                f'sharp clean strokes. '
                f'{GS.STYLE}')
    if lang == '日语':
        chars = ' '.join(text)
        return (f'A photorealistic cinematic background of {en}. '
                f'The scene is empty of people. '
                f'A large simple rectangular signboard with big bold characters fills a clear area of the scene. '
                f'The sign displays Japanese katakana/kanji written EXACTLY as: "{text}". '
                f'Each glyph is clearly separated and must appear in this exact order: {chars}. '
                f'Correct Japanese glyph shapes, no substitution, legible and sharp. '
                f'{GS.STYLE}')
    if lang == '韩语':
        chars = ' '.join(text)
        return (f'A photorealistic cinematic background of {en}. '
                f'The scene is empty of people. '
                f'A large simple rectangular signboard with big bold characters fills a clear area of the scene. '
                f'The sign displays Korean Hangul written EXACTLY as: "{text}". '
                f'Each syllable block is clearly separated and must appear in this exact order: {chars}. '
                f'Correct Hangul block shapes, no substitution, legible and sharp. '
                f'{GS.STYLE}')
    return (f'A photorealistic cinematic background of {en}. '
            f'The scene is empty of people. '
            f'A large simple rectangular signboard with big bold characters fills a clear area of the scene. '
            f'The sign displays exactly this text: "{text}". '
            f'Correct, legible, sharp strokes on a realistic signboard. '
            f'{GS.STYLE}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sids', required=True)
    ap.add_argument('--url', required=True)
    ap.add_argument('--tries', type=int, default=4)
    ap.add_argument('--seed-off', type=int, default=0, help='种子偏移，换空间避免复现坏图')
    args = ap.parse_args()

    os.environ['COMFY_URL'] = args.url
    importlib.reload(comfy)

    src = open('/ssd2/duanju/scripts/gen_scenes_100.py', encoding='utf-8').read()
    ns = {}
    m = re.search(r"SCENES = \[(.*?)\n\]", src, re.S)
    exec("SCENES = [" + m.group(1) + "\n]", ns)
    SCENES = {s[0]: s for s in ns['SCENES']}

    sids = [x.strip() for x in args.sids.split(',') if x.strip()]
    for sid in sids:
        if sid not in SCENES:
            print(f'skip unknown {sid}')
            continue
        _, cn, en, text, lang = SCENES[sid]
        if sid in REPLACE:
            newtext, newlang = REPLACE[sid][1], REPLACE[sid][2]
            text = newtext
            lang = newlang
        idx = list(SCENES.keys()).index(sid)
        # 每次重生成都换全新种子空间，避免复现上一轮的坏图
        base_seed = 770000 + idx * 37 + 900000 + int(args.seed_off)
        dst = f"{ROOT}/{sid}.png"
        prompt = build_prompt(en, text, lang)
        ok = False
        for attempt in range(args.tries):
            try:
                fs = GS.run_one(GS.up('blank:1216x832'), prompt, base_seed + attempt * 13,
                                f'scene_{sid}', ROOT, GS.NEG)
                if fs:
                    os.replace(fs[0], dst)
                    ok = True
                    print(f'{sid} regenerated seed={base_seed + attempt*13} text={text!r} lang={lang}', flush=True)
                    break
            except Exception as ex:
                print(f'  {sid} attempt {attempt}: {ex}', flush=True)
                time.sleep(3)
        if not ok:
            print(f'{sid} FAILED', flush=True)


if __name__ == '__main__':
    main()
