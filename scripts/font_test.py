#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""字体渲染隔离测试：黑底白字 + 红底金字，直接让 VL 读，判断字形映射是否正确。"""
import os

from PIL import Image, ImageDraw, ImageFont

OUT = '/ssd2/duanju/_font_test'
FONTS = [
    ('SourceHanSansSC-Bold.otf', '/usr/share/fonts/truetype/custom/SourceHanSansSC-Bold.otf'),
    ('NotoSansSC-Bold.ttf', '/usr/share/fonts/truetype/custom/NotoSansSC-Bold.ttf'),
]
SAMPLES = ['白鹿书院', '悦来客栈', '正阳殿', '公用电话', '永安百货', '治疗养院']


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, path in FONTS:
        try:
            f = ImageFont.truetype(path, 64)
        except Exception as ex:
            print(f'{name}: 加载失败 {ex}')
            continue
        img = Image.new('RGB', (900, 140 + 80 * len(SAMPLES)), (18, 18, 22))
        d = ImageDraw.Draw(img)
        d.text((20, 10), name, font=ImageFont.truetype(path, 22), fill=(120, 220, 255))
        y = 50
        for s in SAMPLES:
            d.text((20, y), s, font=f, fill=(245, 245, 240))
            # 同字号下画出 .notdef 对照
            d.text((520, y), chr(0xE000), font=f, fill=(255, 90, 90))
            y += 80
        dst = f'{OUT}/{name.replace(".", "_")}.png'
        img.save(dst)
        print(f'OK {dst}  (size={img.size})')


if __name__ == '__main__':
    main()
