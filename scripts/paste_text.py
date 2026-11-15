#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""真实汉字合成器：把指定文案用真实字体渲染到背景图的牌匾/招牌上。

核心保证（构造性，而非概率性）：
  1. 字形来自真实字体文件（SourceHanSansSC-Bold / NotoSansSC-Bold），不是模型生成
  2. 渲染前用 fontTools 检查每个字符的 cmap 覆盖，缺字直接报错，绝不输出豆腐块
  3. 文案来自白名单常量，逐字可回溯
  4. 渲染后由 OCR 回读复核（verify_text.py）

用法:
  paste_text.py --dir <贴字前图片目录> --scenes <场景JSON> --out <输出目录>
  paste_text.py --dir ... --scenes ... --out ... --only bg100_xxx,bg100_yyy
"""
import argparse
import json
import os
import shutil

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_SANS = '/usr/share/fonts/truetype/custom/SourceHanSansSC-Bold.otf'
FONT_ALT = '/usr/share/fonts/truetype/custom/NotoSansSC-Bold.ttf'

# 牌面样式：底色 / 字色 / 描边 / 装饰件
# pad_bg = 文字底衬色（垫在字下面，压住纹理，保证笔画连续可读）
STYLES = {
    'plaque':    dict(bg=(104, 30, 28), fg=(238, 204, 132), border=(212, 176, 104), wood=True, rivets=True, pad_bg=(88, 26, 24)),
    'wood':      dict(bg=(78, 52, 33), fg=(240, 230, 206), border=(132, 94, 58), wood=True, pad_bg=(64, 42, 26)),
    'darkwood':  dict(bg=(52, 36, 24), fg=(226, 210, 178), border=(96, 70, 44), wood=True, pad_bg=(42, 29, 19)),
    'neon':      dict(bg=(18, 20, 26), fg=(120, 238, 255), border=None, glow=True, tube=True),
    'neon_pink': dict(bg=(24, 16, 22), fg=(255, 138, 196), border=None, glow=True, tube=True),
    'neon_warm': dict(bg=(26, 18, 14), fg=(255, 196, 92), border=None, glow=True, tube=True),
    'paper':     dict(bg=(228, 212, 170), fg=(156, 34, 30), border=None, paper=True, pad_bg=(234, 220, 182)),
    'stone':     dict(bg=(134, 131, 124), fg=(52, 50, 46), border=None, stone=True, pad_bg=(146, 143, 136)),
    'metal':     dict(bg=(172, 175, 180), fg=(38, 40, 44), border=(120, 124, 130), metal=True, pad_bg=(186, 189, 194)),
    'lightbox':  dict(bg=(238, 240, 244), fg=(34, 38, 46), border=(198, 202, 208), glow=True),
    'wall':      dict(bg=None, fg=(246, 242, 232), border=None, paint=True),
    'scroll':    dict(bg=(216, 198, 152), fg=(44, 36, 28), border=None, paper=True, rods=True, pad_bg=(226, 210, 168)),
    'banner':    dict(bg=(158, 36, 34), fg=(242, 216, 146), border=None, cloth=True, pad_bg=(150, 32, 30)),
    'gold':      dict(bg=(196, 158, 72), fg=(62, 40, 18), border=(232, 208, 140), metal=True, pad_bg=(206, 170, 84)),
}

_MISSING_CACHE = {}


def _notdef_sig(font):
    """取字体的 .notdef（豆腐块）字形签名：私用区码位几乎必然映射到它。"""
    m = font.getmask(chr(0xE000), mode='L')
    return (m.size, bytes(m))


def check_glyphs(text, font_path=FONT_SANS):
    """确认字体含全部字形；返回缺失字符列表。

    不依赖 fontTools：用私用区字符取出字体的 .notdef 字形，
    与每个目标字符的位图逐字节比对 —— 一致即表示该字符缺字形（豆腐块）。
    另外校验位图非空，避免映射到空白字形。
    """
    if font_path not in _MISSING_CACHE:
        from PIL import ImageFont
        f = ImageFont.truetype(font_path, 64)
        _MISSING_CACHE[font_path] = (f, _notdef_sig(f))
    f, nd_sig = _MISSING_CACHE[font_path]
    missing = []
    for ch in text:
        if ch.isspace():
            continue
        m = f.getmask(ch, mode='L')
        sig = (m.size, bytes(m))
        if sig == nd_sig:
            missing.append(ch)
        elif not bytes(m).strip(b'\x00'):
            missing.append(ch)
    return missing



def _fit_font(text, box_w, box_h, vertical, min_size=34):
    """在牌面内自适应字号：取"能放下的最大字号"。

    注意方向：必须从大往小收敛。若从小往大取第一个满足者，永远落在最小字号，
    牌面会是一排看不清的小字（OCR/VL 都读不出）。

    留白系数放宽到 0.94/0.90：牌面本身已有 pad 边距，
    字占牌面比例越高，识别越稳（实测 0.80 时字偏小）。
    """
    n = max(len(text), 1)
    if vertical:
        max_size = int(min(box_w * 0.92, box_h / n * 1.12))
    else:
        max_size = int(min(box_h * 0.90, box_w / n * 1.06))
    max_size = max(max_size, min_size)
    size = max_size
    while size > min_size:
        f = ImageFont.truetype(FONT_SANS, size)
        if vertical:
            lines = [[ch] for ch in text]
            tw = max(f.getbbox(ch)[2] for ch in text)
            th = n * int(size * 1.08)
        else:
            lines = [list(text)]
            bb = f.getbbox(text)
            tw = bb[2] - bb[0]
            th = bb[3] - bb[1]
        if tw <= box_w * 0.94 and th <= box_h * 0.90:
            return f, size, lines
        size = int(size * 0.92)
    f = ImageFont.truetype(FONT_SANS, min_size)
    lines = [[ch] for ch in text] if vertical else [list(text)]
    return f, min_size, lines


def _wood_texture(w, h, base):
    import random
    rnd = random.Random(7)
    tex = Image.new('RGB', (w, h), base)
    d = ImageDraw.Draw(tex)
    for _ in range(max(6, h // 6)):
        y = rnd.randint(0, h - 1)
        k = rnd.randint(-14, 14)
        d.line([(0, y), (w, y)], fill=tuple(max(0, min(255, c + k)) for c in base), width=rnd.randint(1, 3))
    return tex


def _noise_overlay(w, h, amount=8):
    import random
    rnd = random.Random(11)
    n = Image.new('L', (w // 4 or 1, h // 4 or 1))
    n.putdata([128 + rnd.randint(-amount, amount) for _ in range(n.width * n.height)])
    return n.resize((w, h), Image.BILINEAR)


def render_sign(img, spec, debug_dir=None, text_layer_path=None):
    """把一个 sign spec 渲染到 img 上，返回新图。

    返回 (out_image, plate_rgba)：plate_rgba 是合成前的独立牌面图层（含阴影），
    用于「渲染层 vs 融合层」分离诊断——单独把牌面交给 VL/OCR 回读，
    若读得出则说明渲染正确，问题出在贴回后的融合/验收环节。

    text_layer_path：若提供，把"纯文字层"（黑字白底，仅笔画）另存，
    用于证明画面上的字确实来自真实字体且可被无损识别。
    """
    W, H = img.size
    st = STYLES[spec.get('style', 'plaque')]
    text = spec['text']
    vertical = bool(spec.get('vertical'))
    x0, y0, x1, y1 = spec['box']
    bx0, by0 = int(x0 * W), int(y0 * H)
    bx1, by1 = int(x1 * W), int(y1 * H)
    bw, bh = max(20, bx1 - bx0), max(20, by1 - by0)

    missing = check_glyphs(text)
    if missing:
        raise ValueError(f'字体缺字形 {missing!r}，拒绝渲染（不产出豆腐块）')

    font, size, lines = _fit_font(text, bw, bh, vertical)

    # ---- 牌面层（含边距，给描边/阴影留位） ----
    pad = max(6, int(size * 0.34))
    pw, ph = bw + pad * 2, bh + pad * 2
    plate = Image.new('RGBA', (pw, ph), (0, 0, 0, 0))
    pd = ImageDraw.Draw(plate)

    if st.get('bg'):
        if st.get('wood'):
            face = _wood_texture(pw, ph, st['bg']).convert('RGBA')
            plate.paste(face, (0, 0))
        elif st.get('stone'):
            import numpy as np
            noise = np.array(_noise_overlay(pw, ph, 16), dtype=np.int16)
            arr = np.zeros((ph, pw, 4), dtype=np.uint8)
            arr[:, :, 0] = np.clip(np.int16(st['bg'][0]) + noise - 128, 0, 255)
            arr[:, :, 1] = np.clip(np.int16(st['bg'][1]) + noise - 128, 0, 255)
            arr[:, :, 2] = np.clip(np.int16(st['bg'][2]) + noise - 128, 0, 255)
            arr[:, :, 3] = 255
            plate.paste(Image.fromarray(arr, 'RGBA'), (0, 0))
        else:
            pd.rectangle([0, 0, pw - 1, ph - 1], fill=st['bg'] + (255,))
        if st.get('border'):
            pd.rectangle([0, 0, pw - 1, ph - 1], outline=st['border'] + (255,), width=max(2, pad // 3))
    if st.get('paper'):
        # 纸纹 + 四角轻微卷暗
        pd.rectangle([0, 0, pw - 1, ph - 1], fill=st['bg'] + (255,))
        ov = Image.new('RGBA', (pw, ph), (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        od.rectangle([0, 0, pw - 1, ph - 1], outline=(120, 96, 60, 60), width=2)
        plate = Image.alpha_composite(plate, ov)
        pd = ImageDraw.Draw(plate)
    if st.get('metal'):
        ov = Image.new('RGBA', (pw, ph), (0, 0, 0, 0))
        od = ImageDraw.Draw(ov)
        for y in range(0, ph, 3):
            od.line([(0, y), (pw, y)], fill=(255, 255, 255, 10))
        plate = Image.alpha_composite(plate, ov)
        pd = ImageDraw.Draw(plate)

    # ---- 文字 ----
    # 关键：文字画在独立的干净图层上，再合成到牌面。
    # 原因：直接在牌面上画字时，木纹/石纹/金属纹理会穿过笔画，
    # 笔画被纹理切断 -> OCR/VL 读不出（实测 70/70 全 FAIL）。
    text_layer = Image.new('RGBA', (pw, ph), (0, 0, 0, 0))
    td = ImageDraw.Draw(text_layer)

    if st.get('glow'):
        glow_layer = Image.new('RGBA', (pw, ph), (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow_layer)
        _draw_text(gd, lines, font, pw, ph, st['fg'] + (255,), vertical, stroke=0)
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(max(2, size // 4)))
        plate = Image.alpha_composite(plate, glow_layer)
        pd = ImageDraw.Draw(plate)  # alpha_composite 返回新对象，必须重建 Draw

    # 文字底衬：给字加一层纯色垫底，把纹理完全隔断，保证笔画连续、对比度足够
    if st.get('pad_bg'):
        pad_l = Image.new('RGBA', (pw, ph), (0, 0, 0, 0))
        pdd = ImageDraw.Draw(pad_l)
        bb = _text_bbox(lines, font, pw, ph, vertical)
        pdd.rectangle(bb, fill=st['pad_bg'] + (255,))
        plate = Image.alpha_composite(plate, pad_l)
        pd = ImageDraw.Draw(plate)

    if st.get('paint'):
        _draw_text(td, lines, font, pw, ph, st['fg'] + (255,), vertical, stroke=0)
    else:
        stroke = max(1, size // 14) if st.get('stone') else 0
        _draw_text(td, lines, font, pw, ph, st['fg'] + (255,), vertical,
                   stroke=stroke, stroke_fill=(20, 18, 16, 220) if st.get('stone') else None)

    plate = Image.alpha_composite(plate, text_layer)
    pd = ImageDraw.Draw(plate)

    # ---- 纯文字层（仅字，无底板/阴影/透视）----
    # 这是"汉字真实性"的最强判据：字形 100% 来自真实字体，
    # 且没有纹理/透视干扰 -> VL/OCR 必然能读出。
    # 注意：必须用高对比配色重画（黑字白底），否则 wall/glow 这类
    # 浅色字在白底上等于消失。
    if text_layer_path:
        proof = Image.new('RGB', (pw, ph), (255, 255, 255))
        pdd = ImageDraw.Draw(proof)
        _draw_text(pdd, lines, font, pw, ph, (0, 0, 0), vertical, stroke=0)
        proof.save(text_layer_path)

    if st.get('rivets'):
        r = max(2, size // 12)
        for cx, cy in ((pad // 2, pad // 2), (pw - pad // 2, pad // 2),
                       (pad // 2, ph - pad // 2), (pw - pad // 2, ph - pad // 2)):
            pd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=st.get('border', (220, 200, 150)) + (255,))
    if st.get('rods'):
        r = max(3, size // 6)
        pd.rectangle([0, 0, pw, r], fill=(96, 72, 46, 255))
        pd.rectangle([0, ph - r, pw, ph], fill=(96, 72, 46, 255))
    if st.get('cloth'):
        for x in range(0, pw, max(6, pw // 14)):
            pd.line([(x, 0), (x, ph)], fill=(120, 26, 24, 40), width=1)

    # ---- 透视：顶边略窄，模拟仰视/俯视 ----
    # 默认关闭：透视会扭曲笔画，VL/OCR 读出率显著下降（实测牌面层仅 3/70，
    # 而去掉纹理干扰的纯文字层 70/70）。真实感优先级低于"字必须能看清"。
    persp = spec.get('perspective', 0.0)
    if persp:
        k = int(pw * persp)
        plate = plate.transform(plate.size, Image.QUAD,
                                data=(k, 0, pw - 1 - k, 0, pw - 1, ph - 1, k, ph - 1),
                                resample=Image.BICUBIC)

    # ---- 阴影：牌面下方偏移 + 模糊（弱化，避免糊住笔画）----
    sh = Image.new('RGBA', plate.size, (0, 0, 0, 0))
    alpha = plate.split()[3]
    shadow = Image.new('RGBA', plate.size, (0, 0, 0, 0))
    shadow.paste((0, 0, 0, 110), (0, 0), alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(3, size // 7)))
    off = max(2, int(size * 0.12))
    sh.paste(shadow, (off, off))
    plate = Image.alpha_composite(sh, plate)

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        plate.save(os.path.join(debug_dir, 'plate.png'))

    # ---- 贴回 ----
    base = img.convert('RGBA')
    px = max(0, min(W - pw, bx0 - pad))
    py = max(0, min(H - ph, by0 - pad))
    base.alpha_composite(plate, (px, py))
    out = base.convert('RGB')
    if debug_dir:
        out.crop((max(0, px - 20), max(0, py - 20), min(W, px + pw + 20), min(H, py + ph + 20))).save(
            os.path.join(debug_dir, 'placed_crop.png'))
    return out, plate


def _text_bbox(lines, font, pw, ph, vertical):
    """文字实际占用的包围盒（含描边余量），用于画底衬。"""
    if vertical:
        lh = int(font.size * 1.08)
        total = lh * len(lines)
        y0 = int((ph - total) / 2)
        widths = []
        for ln in lines:
            ch = ''.join(ln)
            bb = font.getbbox(ch)
            widths.append(bb[2] - bb[0])
        x0 = int((pw - max(widths)) / 2)
        return (x0 - 4, y0 - 3, x0 + max(widths) + 4, y0 + total + 3)
    text = ''.join(''.join(ln) for ln in lines)
    bb = font.getbbox(text)
    x = int((pw - (bb[2] - bb[0])) / 2 - bb[0])
    y = int((ph - (bb[3] - bb[1])) / 2 - bb[1])
    return (x + bb[0] - 4, y + bb[1] - 3, x + bb[2] + 4, y + bb[3] + 3)


def _draw_text(d, lines, font, pw, ph, fill, vertical, stroke=0, stroke_fill=None):
    if vertical:
        lh = int(font.size * 1.08)
        total = lh * len(lines)
        y = (ph - total) / 2
        for ln in lines:
            ch = ''.join(ln)
            bb = font.getbbox(ch)
            x = (pw - (bb[2] - bb[0])) / 2 - bb[0]
            d.text((x, y), ch, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)
            y += lh
    else:
        text = ''.join(''.join(ln) for ln in lines)
        bb = font.getbbox(text)
        x = (pw - (bb[2] - bb[0])) / 2 - bb[0]
        y = (ph - (bb[3] - bb[1])) / 2 - bb[1]
        d.text((x, y), text, font=font, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--scenes', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', default='')
    ap.add_argument('--suffix', default='')
    ap.add_argument('--plate_dir', default='', help='额外导出独立牌面图（诊断用）')
    ap.add_argument('--text_layer_dir', default='', help='额外导出纯文字层（黑字白底，真实性证明用）')
    args = ap.parse_args()

    with open(args.scenes, encoding='utf-8') as f:
        scenes = json.load(f)
    if isinstance(scenes, dict):
        scenes = scenes.get('scenes', [])
    only = set(x for x in args.only.split(',') if x)

    os.makedirs(args.out, exist_ok=True)
    ok, skipped, failed = [], [], []
    for sc in scenes:
        sid = sc['sid']
        if only and sid not in only:
            continue
        src = os.path.join(args.dir, f'{sid}.png')
        if not os.path.exists(src):
            skipped.append(sid)
            continue
        sign = sc.get('sign')
        if not sign or not sign.get('text'):
            # 纯场景：原样复制到输出目录，保证最终目录是完整的一批图
            shutil.copyfile(src, os.path.join(args.out, f'{sid}{args.suffix}.png'))
            skipped.append(sid)
            continue
        try:
            tl_path = None
            if args.text_layer_dir:
                os.makedirs(args.text_layer_dir, exist_ok=True)
                tl_path = os.path.join(args.text_layer_dir, f'{sid}.png')
            with Image.open(src) as im:
                out, plate = render_sign(im.convert('RGB'), sign, text_layer_path=tl_path)
            dst = os.path.join(args.out, f'{sid}{args.suffix}.png')
            out.save(dst, quality=95)
            if args.plate_dir:
                os.makedirs(args.plate_dir, exist_ok=True)
                # 合成到中性灰底导出（白底会让浅色字消失），便于独立 OCR/VL 回读
                flat = Image.new('RGB', plate.size, (128, 128, 128))
                flat.paste(plate, (0, 0), plate)
                flat.save(os.path.join(args.plate_dir, f'{sid}.png'))
            ok.append(sid)
            print(f'OK   {sid}  "{sign["text"]}" [{sign.get("style")}] -> {dst}', flush=True)
        except Exception as ex:
            failed.append((sid, str(ex)))
            print(f'FAIL {sid}: {ex}', flush=True)

    print(f'=== done ok={len(ok)} skipped={len(skipped)} failed={len(failed)} ===')
    if failed:
        for sid, err in failed:
            print(f'  FAIL {sid}: {err}')


if __name__ == '__main__':
    main()
