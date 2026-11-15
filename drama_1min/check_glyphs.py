#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字形预检（glyph preflight）— 严禁豆腐块。

检查字体 cmap 是否覆盖字幕/画面中出现的每一个汉字。
用法：check_glyphs.py <字体路径> <文本文件...>
"""
import sys
from fontTools.ttLib import TTFont, TTCollection


def load_cmap(path):
    if path.lower().endswith(".ttc") or path.lower().endswith(".otc"):
        coll = TTCollection(path)
        return set(coll.fonts[0].getBestCmap().keys())
    return set(TTFont(path, fontNumber=0).getBestCmap().keys())


def main():
    font_path = sys.argv[1]
    cmap = load_cmap(font_path)
    print(f"字体: {font_path}")
    print(f"cmap 码点数: {len(cmap)}")

    missing_all = set()
    for txt_path in sys.argv[2:]:
        text = open(txt_path, encoding="utf-8").read()
        chars = {c for c in text if not c.isspace()}
        missing = sorted(c for c in chars if ord(c) not in cmap)
        print(f"\n[{txt_path}] 字符数={len(chars)} 缺字={len(missing)}")
        if missing:
            print("  缺失:", "".join(missing))
            missing_all.update(missing)
        else:
            print("  ✓ 全部覆盖")

    if missing_all:
        print(f"\n❌ 共 {len(missing_all)} 个缺字: {''.join(sorted(missing_all))}")
        return 1
    print("\n✅ 字形预检通过，无豆腐块风险")
    return 0


if __name__ == "__main__":
    sys.exit(main())
