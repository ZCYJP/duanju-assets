#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《决裂》15秒样本 — 四层字幕渲染（PIL）

严格按「短剧语法规范（参考片实测基准）」§4：
  A 顶部信息层  y=0.080  字高 16~30px   剧名+集数
  C 中文主字幕  y=0.686  字高 画面高4.1% (=34px)  台词（不带「角色：」前缀）
  D 英文副字幕  y=0.822  字高 画面高3.8% (=31px)  英文翻译
  E 底部固定层  y=0.966  字高 12~25px   频道水印 + AI生成声明

字号自适应：二分搜索 font size 使墨迹高度命中目标，且宽度不溢出画面 86%。
输出：subs/{top,bot,cn_XX,en_XX}.png（480x832 RGBA）
"""
import os
import sys
import json
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shots_15s import SHOTS, EN_SUB, TOP_BAR, FPS

BASE = "/ssd2/duanju/drama_1min/sample_15s"
SUB_DIR = f"{BASE}/subs"
os.makedirs(SUB_DIR, exist_ok=True)

FONT_PATH = "/usr/share/fonts/truetype/custom/SourceHanSansSC-Bold.otf"
W, H = 480, 832

# 规范位置（比例）
Y_TOP = 0.080
Y_CN = 0.686
Y_EN = 0.822
Y_BOT = 0.966
SZ_TOP = 22      # 16~30px
SZ_BOT = 16      # 12~25px
SZ_CN = 34       # 画面高 4.1%
SZ_EN = 31       # 画面高 3.8%
MAX_W_RATIO = 0.86

# 四层清单（y比例, 目标字高px, 颜色, 描边宽度）
LAYERS = {
    "top": (Y_TOP, SZ_TOP, (235, 235, 235, 225), 2),
    "bot": (Y_BOT, SZ_BOT, (190, 190, 190, 205), 2),
    "cn":  (Y_CN, SZ_CN, (255, 255, 255, 255), 4),
    "en":  (Y_EN, SZ_EN, (225, 225, 225, 245), 3),
}

_font_cache = {}


def get_font(size):
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(FONT_PATH, size)
    return _font_cache[size]


def text_ink_size(font, text):
    """测量墨迹实际宽高"""
    tmp = Image.new("L", (10, 10))
    d = ImageDraw.Draw(tmp)
    bb = d.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0], bb[3] - bb[1], bb


def fit_font(text, target_h, max_w):
    """二分字号：墨迹高命中 target_h，且宽 <= max_w"""
    lo, hi = 8, 120
    best = get_font(10)
    while lo <= hi:
        mid = (lo + hi) // 2
        f = get_font(mid)
        w, h, _ = text_ink_size(f, text)
        if h <= target_h and w <= max_w:
            best = f
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def render_layer(text, y_ratio, target_h, color, stroke, path, max_ratio=MAX_W_RATIO):
    max_w = W * max_ratio
    font = fit_font(text, target_h, max_w)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    w, h, bb = text_ink_size(font, text)
    y_px = y_ratio * H
    # anchor 用 mm（水平居中，垂直居中于该层基准线）
    d.text((W / 2, y_px), text, font=font, fill=color, anchor="mm",
           stroke_width=stroke, stroke_fill=(0, 0, 0, 220))
    img.save(path)
    return {"text": text, "font_px": font.size, "ink_h": h, "ink_w": w,
            "y_px": round(y_px, 1), "path": os.path.basename(path)}


def check_glyphs(chars):
    """字形预检：优先 fontTools cmap，退化用 fontconfig :charset 查询"""
    chars = set(chars)
    try:
        from fontTools.ttLib import TTFont
        cmap = set(TTFont(FONT_PATH).getBestCmap().keys())
        missing = sorted(c for c in chars if ord(c) not in cmap)
        return missing, "fontTools cmap"
    except ImportError:
        pass
    import subprocess
    miss = []
    for c in chars:
        r = subprocess.run(["fc-list", ":charset=%x" % ord(c), "file"],
                           capture_output=True, text=True)
        if FONT_PATH not in r.stdout:
            miss.append(c)
    return sorted(miss), "fontconfig charset"


def main():
    # 字体字形预检（cmap 含简体字，防豆腐块）
    need = set()
    for s in SHOTS:
        need |= set(s["dialogue"][0][1])
    need |= set("".join(EN_SUB[s["name"]] for s in SHOTS))
    need |= set(TOP_BAR["title"]) | set(TOP_BAR["ep"]) | set("AI生成纯属虚构请勿模仿")
    missing, how = check_glyphs(need)
    if missing:
        print("字体缺字形(%s): %s" % (how, missing))
        raise SystemExit("字体 cmap 预检失败")
    print("字体字形预检通过（%s）：%d 字全部可渲染" % (how, len(need)))

    report = {"layers": {}, "cn": {}, "en": {}}
    # A 顶部信息层
    top_txt = "%s · %s" % (TOP_BAR["title"], TOP_BAR["ep"])
    report["layers"]["top"] = render_layer(top_txt, *LAYERS["top"], f"{SUB_DIR}/top.png")
    # E 底部固定层
    bot_txt = "AI生成 · 纯属虚构 · 请勿模仿"
    report["layers"]["bot"] = render_layer(bot_txt, *LAYERS["bot"], f"{SUB_DIR}/bot.png")

    for s in SHOTS:
        sid = s["id"]
        cn = s["dialogue"][0][1]
        en = EN_SUB[s["name"]]
        r1 = render_layer(cn, *LAYERS["cn"], f"{SUB_DIR}/cn_{sid:02d}.png")
        r2 = render_layer(en, *LAYERS["en"], f"{SUB_DIR}/en_{sid:02d}.png")
        report["cn"][str(sid)] = r1
        report["en"][str(sid)] = r2
        print("[%d] CN %-22s %2dpx 高 %3dpx 宽 | EN %-40s %2dpx" %
              (sid, cn, r1["font_px"], r1["ink_w"], en, r2["font_px"]), flush=True)

    json.dump(report, open(f"{BASE}/logs/sub_report.json", "w"), ensure_ascii=False, indent=2)
    print("\n字幕层渲染完成 →", SUB_DIR)


if __name__ == "__main__":
    main()
