#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""文字核验规则 v5 —— 修正版（字符规则 + 标签解析歧义），以 monkeypatch 注入。

修正一：字符集
  原规则只允许「ASCII 字母数字」+「CJK/假名/谚文」，导致两类合法文字被误判：
  1. 空白字符（空格 U+0020 / 制表 U+0009 / 全角空格 U+3000）
     多词招牌的词间空格不是伪文字。
     证据：bg_driving_office 预期 "DRIVE SAFE"，easyocr 四个 reader 全部正确读出
     "DRIVE SAFE"，却因 bad_chars=[' '], bad_codepoints=['U+0020'] 被剔除，
     最终图像级判定误判为「伪文字」。
  2. 拉丁扩展字母（U+00C0-U+024F）
     法语 é / è / ç 等重音字母不在允许集内，法语招牌同样被误判。

修正二：标签解析歧义（parse_tagged）
  VL 放大层在无法确定时返回候选标签列表，例如：
      "石磨豆腐 | [REAL]/[FAKE]/[CJK]/[无法辨认]"
  这是「请从这些标签里选」的提示语，而不是「同时命中 REAL 和 FAKE」。
  原实现用 re.findall 提取全部方括号标签，只要含 [FAKE] 就判伪，
  导致 bg_tofu_stall 这类实际完全正确的招牌被误判为「伪文字」。

  修正：解析前先判断右侧是否为「候选标签列表」——若去掉空白后形如
  连续的 [X]/[Y]/[Z]（>=2 个标签用 / 连接），则不提取任何标签，
  视为「未给出标签」，不作为伪文字证据。

用法
----
    import tmp_text_rule_v5 as R
    R.apply()             # patch tmp_bg_text_audit
    R.apply_adjudicate()  # patch tmp_bg_text_adjudicate
"""
import os
import re
import sys

SCRIPTS = "/ssd2/duanju/scripts"
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

LATIN_EXT = (0x00C0, 0x024F)   # 拉丁扩展字母


def _char_allowed(ch):
    cp = ord(ch)
    if ch.isspace():
        return True
    if ch.isascii() and ch.isalnum():
        return True
    if LATIN_EXT[0] <= cp <= LATIN_EXT[1]:
        return True
    return False


def apply():
    """patch tmp_bg_text_audit.classify_chars / char_detail。"""
    import tmp_bg_text_audit as A

    def classify_chars(s):
        stripped = s.strip()
        if not stripped:
            return "empty", [], []
        bad = []
        for ch in stripped:
            if _char_allowed(ch):
                continue
            if A._in_ranges(ord(ch)):
                continue
            bad.append(ch)
        return ("pseudo" if bad else "real"), bad, [f"U+{ord(c):04X}" for c in bad]

    def char_detail(s):
        out = []
        for ch in s:
            cp = ord(ch)
            if ch.isspace():
                k = "space"
            elif ch.isascii() and ch.isalnum():
                k = "ascii_alnum"
            elif LATIN_EXT[0] <= cp <= LATIN_EXT[1]:
                k = "latin_ext"
            elif A._in_ranges(cp):
                k = "cjk_or_kana_or_hangul"
            else:
                k = "other"
            out.append((ch, k))
        return out

    A.classify_chars = classify_chars
    A.char_detail = char_detail
    return A


# 候选标签列表形如 [REAL]/[FAKE]/[CJK]/[无法辨认]
_CANDIDATE_LIST = re.compile(r"^\s*\[[^\]]+\](?:\s*/\s*\[[^\]]+\])+\s*$")


def _parse_tagged_fixed(lines):
    """修正版标签解析：识别候选标签列表，不把它当作已选定标签。"""
    out = []
    for l in lines:
        if "|" not in l:
            continue
        left, right = l.split("|", 1)
        t = left.strip()
        if not t:
            continue
        if _CANDIDATE_LIST.match(right):
            # 候选列表 -> 未给出确定标签，不作为任何证据
            out.append((t, []))
            continue
        tags = re.findall(r"\[[^\]]+\]", right)
        out.append((t, [x.upper() for x in tags]))
    return out


def apply_adjudicate():
    """patch tmp_bg_text_adjudicate.char_ok / parse_tagged。"""
    import tmp_bg_text_adjudicate as J

    def char_ok(ch):
        cp = ord(ch)
        if _char_allowed(ch):
            return True
        return any(lo <= cp <= hi for lo, hi in J.ALLOWED_RANGES)

    J.char_ok = char_ok
    J.parse_tagged = _parse_tagged_fixed
    return J


SELF_CHECK_CHARS = [
    ("DRIVE SAFE", "real", "英文多词"),
    ("Salon de Beauté", "real", "法语重音+空格"),
    ("発掘 現場", "real", "日文+空格"),
    ("출장 서비스", "real", "韩文+空格"),
    ("出发大厅", "real", "简体中文"),
    ("家 政 服 務", "real", "繁体+空格"),
    ("EXAACT", "real", "合法 ASCII（语义层兜底）"),
    ("FABLISATEBIC", "real", "合法 ASCII（语义层兜底）"),
    ("DRIWE SAFE", "real", "合法 ASCII（语义层兜底）"),
    ("ZBR", "real", "合法 ASCII"),
    ("FRUNIC /AOI TYEFA", "pseudo", "含斜杠"),
    ("宠[伪文字]物店", "pseudo", "含方括号"),
    ("|-一", "pseudo", "含竖线/连字符"),
    ("Right — 磨", "pseudo", "含破折号"),
]

SELF_CHECK_TAGS = [
    (["石磨豆腐 | [REAL]/[FAKE]/[CJK]/[无法辨认]"], "石磨豆腐", [], "候选标签列表->无标签"),
    (["诚信律师事務所 | [REAL]"], "诚信律师事務所", ["[REAL]"], "单一标签"),
    (["宠[伪文字]物店 | [FAKE]"], "宠[伪文字]物店", ["[FAKE]"], "明确FAKE"),
    (["X | [伪文字]"], "X", ["[伪文字]"], "中文伪标记"),
]


def self_check():
    A = apply()
    J = apply_adjudicate()
    fail = 0
    print("=== 规则 v5 自检：字符集 ===")
    for text, want, note in SELF_CHECK_CHARS:
        got, bad, cps = A.classify_chars(text)
        ok = got == want
        fail += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {text!r:28s} -> {got:6s} (want {want})  {note}")

    print("=== 规则 v5 自检：标签解析 ===")
    for lines, want_text, want_tags, note in SELF_CHECK_TAGS:
        parsed = J.parse_tagged(lines)
        ok = len(parsed) == 1 and parsed[0][0] == want_text and parsed[0][1] == want_tags
        fail += 0 if ok else 1
        print(f"  {'PASS' if ok else 'FAIL'}  {str(lines[0])[:46]:48s} -> {parsed}  {note}")

    total = len(SELF_CHECK_CHARS) + len(SELF_CHECK_TAGS)
    print(f"=== {total-fail}/{total} 通过 ===")
    return fail


if __name__ == "__main__":
    sys.exit(1 if self_check() else 0)
