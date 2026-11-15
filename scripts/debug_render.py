#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""隔离测试：在纯色底图上渲染牌匾，导出 plate 图层与合成结果。"""
import os
import sys

sys.path.insert(0, '/ssd2/duanju/scripts')
from PIL import Image  # noqa: E402
import paste_text  # noqa: E402

OUT = '/ssd2/duanju/_render_debug'
CASES = [
    ('plaque', '白鹿书院', [0.28, 0.12, 0.68, 0.23]),
    ('metal', '悦来客栈', [0.26, 0.11, 0.66, 0.22]),
    ('paper', '正阳殿', [0.30, 0.08, 0.70, 0.19]),
    ('lightbox', '公用电话', [0.32, 0.26, 0.66, 0.37]),
    ('stone', '万剑冢', [0.11, 0.46, 0.27, 0.86]),
]


def main():
    os.makedirs(OUT, exist_ok=True)
    for style, text, box in CASES:
        d = os.path.join(OUT, style)
        img = Image.new('RGB', (752, 1392), (128, 128, 132))
        out = paste_text.render_sign(img, {'text': text, 'style': style, 'box': box,
                                           'vertical': style in ('wood', 'darkwood', 'stone')},
                                     debug_dir=d)
        out.save(os.path.join(d, 'full.png'))
        print(f'{style}: plate={os.path.join(d, "plate.png")} crop={os.path.join(d, "placed_crop.png")}')


if __name__ == '__main__':
    main()
