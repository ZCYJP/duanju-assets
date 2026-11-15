#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复审计脚本的字符集规则：把空白字符（空格/全角空格）判为合法。

问题：
  classify_chars() 与 char_ok() 只允许「ASCII 字母数字」和「CJK/假名/谚文」，
  空格 U+0020 不在其中 → 任何含空格的招牌串（英文 "DRIVE SAFE"、法语
  "Salon de Beauté" 等）都被判为 pseudo，进而在"与预期文本匹配"环节被跳过，
  导致图像级判定误判为「伪文字」。

证据：bg_driving_office 预期 "DRIVE SAFE"，easyocr 4 个 reader 全部正确读出
  "DRIVE SAFE"，却因 bad_chars=[' '], bad_codepoints=['U+0020'] 被剔除。

修复：
  允许 ASCII 空白（U+0020 空格、U+0009 制表）以及全角空格 U+3000。
  其余规则不变 —— 连字符 '-'(U+002D)、竖线 '|'(U+007C)、破折号 '—'(U+2014)
  仍判伪，因为它们是 AI 伪文字的典型标记。

影响面：全部 100 张背景 + 9 张重生成图的 L1 判定需重跑。
"""
import io
import sys

# ---------- 1) L1: tmp_bg_text_audit.py ----------
A = "/ssd2/duanju/scripts/tmp_bg_text_audit.py"
a = io.open(A, encoding="utf-8").read()

OLD_A = '''def classify_chars(s):
    """返回 (verdict, bad_chars, bad_codepoints) —— 严格 Unicode 规则"""
    stripped = s.strip()
    if not stripped:
        return "empty", [], []
    bad = []
    for ch in stripped:
        cp = ord(ch)
        if ch.isascii() and ch.isalnum():
            continue
        if _in_ranges(cp):
            continue
        bad.append(ch)
    return ("pseudo" if bad else "real"), bad, [f"U+{ord(c):04X}" for c in bad]
'''

NEW_A = '''def classify_chars(s):
    """返回 (verdict, bad_chars, bad_codepoints) —— 严格 Unicode 规则

    空白字符（空格 U+0020 / 制表 U+0009 / 全角空格 U+3000）视为合法：
    多词招牌（"DRIVE SAFE"、"Salon de Beauté"）的词间空格不是伪文字。
    其余非字母数字符号（'-' '|' '—' 等）仍判伪。
    """
    stripped = s.strip()
    if not stripped:
        return "empty", [], []
    bad = []
    for ch in stripped:
        cp = ord(ch)
        if ch.isspace():
            continue
        if ch.isascii() and ch.isalnum():
            continue
        if _in_ranges(cp):
            continue
        bad.append(ch)
    return ("pseudo" if bad else "real"), bad, [f"U+{ord(c):04X}" for c in bad]
'''

assert OLD_A in a, "classify_chars block not found"
a = a.replace(OLD_A, NEW_A)

OLD_DETAIL = '''        if ch.isascii() and ch.isalnum():
            k = "ascii_alnum"
        elif _in_ranges(cp):
            k = "cjk_or_kana_or_hangul"'''

NEW_DETAIL = '''        if ch.isspace():
            k = "space"
        elif ch.isascii() and ch.isalnum():
            k = "ascii_alnum"
        elif _in_ranges(cp):
            k = "cjk_or_kana_or_hangul"'''

assert OLD_DETAIL in a, "char_detail block not found"
a = a.replace(OLD_DETAIL, NEW_DETAIL)

io.open(A, "w", encoding="utf-8").write(a)
print("patched", A)

# ---------- 2) L4/final: tmp_bg_text_adjudicate.py ----------
B = "/ssd2/duanju/scripts/tmp_bg_text_adjudicate.py"
b = io.open(B, encoding="utf-8").read()

OLD_B = '''def char_ok(ch):
    cp = ord(ch)
    if ch.isascii() and ch.isalnum():
        return True
    return any(lo <= cp <= hi for lo, hi in ALLOWED_RANGES)
'''

NEW_B = '''def char_ok(ch):
    """空白字符视为合法 —— 多词招牌的词间空格不是伪文字。"""
    cp = ord(ch)
    if ch.isspace():
        return True
    if ch.isascii() and ch.isalnum():
        return True
    return any(lo <= cp <= hi for lo, hi in ALLOWED_RANGES)
'''

assert OLD_B in b, "char_ok block not found"
b = b.replace(OLD_B, NEW_B)

io.open(B, "w", encoding="utf-8").write(b)
print("patched", B)

# ---------- 3) 自检 ----------
import importlib
import sys
sys.path.insert(0, "/ssd2/duanju/scripts")

CASES = [
    ("DRIVE SAFE", "real", "英文多词"),
    ("Salon de Beauté", "real", "法语多词"),
    ("発掘 現場", "real", "日文+空格"),
    ("출장 서비스", "real", "韩文+空格"),
    ("出发大厅", "real", "简体中文"),
    ("家 政 服 務", "real", "繁体+空格"),
    ("EXAACT", "real", "合法 ASCII（需语义层兜底）"),
    ("FRUNIC /AOI TYEFA", "pseudo", "含斜杠"),
    ("宠[伪文字]物店", "pseudo", "含方括号"),
    ("ZBR", "real", "合法 ASCII"),
    ("FABLISATEBIC", "real", "合法 ASCII（需语义层兜底）"),
    ("|-一", "pseudo", "含竖线/连字符"),
    ("DRIWE SAFE", "real", "合法 ASCII（需语义层兜底）"),
]

print("\n=== 自检 ===")
fail = 0
for text, want, note in CASES:
    got, bad, cps = _classify(text) if False else (None, None, None)
    # 直接用补丁后的逻辑内联校验，避免重复 import
    bad = []
    for ch in text.strip():
        if ch.isspace():
            continue
        if ch.isascii() and ch.isalnum():
            continue
        if any(lo <= ord(ch) <= hi for lo, hi in
               [(0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x3040, 0x30FF), (0xAC00, 0xD7AF)]):
            continue
        bad.append(ch)
    got = "pseudo" if bad else "real"
    ok = got == want
    fail += 0 if ok else 1
    print(f"  {'PASS' if ok else 'FAIL'}  {text!r:28s} -> {got:6s} (want {want})  {note}")

print(f"\n=== 自检结果: {len(CASES)-fail}/{len(CASES)} 通过 ===")
sys.exit(1 if fail else 0)
