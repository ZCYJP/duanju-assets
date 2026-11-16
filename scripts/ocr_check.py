#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""easyocr 全图扫描：独立于 VL 的第二条判据通道。

VL 是生成式模型，可能整图尺度下把霓虹/纹理误读成"文字"。
easyocr 是纯检测+识别 pipeline，没有"看到像字的纹理就脑补"的倾向，
因此用它交叉验证：VL 说有伪字、easyocr 也检出 -> 实锤；
VL 说有、easyocr 无 -> 大概率是 VL 幻觉。

用法: ocr_check.py --imgs a.png b.png --out /path/ocr.json
"""
import argparse
import gc
import json
import os

os.environ.setdefault('OMP_THREAD_LIMIT', '1')

import easyocr  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imgs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gpu', default='1')
    ap.add_argument('--langs', default='ch_sim,en')
    args = ap.parse_args()

    gpu = args.gpu not in ('', '-1', 'cpu', 'none')
    langs = [x for x in args.langs.split(',') if x]
    reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
    out = []
    for p in args.imgs:
        with Image.open(p) as im:
            arr = np.array(im.convert('RGB'))
        # 整图 + 上下/左右半图，四个视角都扫
        H, W = arr.shape[:2]
        views = {
            'full': arr,
            'top': arr[: H // 2],
            'bottom': arr[H // 2:],
            'left': arr[:, : W // 2],
            'right': arr[:, W // 2:],
        }
        rec = {'path': p, 'views': {}}
        all_hits = []
        for vn, v in views.items():
            try:
                res = reader.readtext(v, detail=1, paragraph=False,
                                      text_threshold=0.5, low_text=0.25,
                                      link_threshold=0.3)
            except Exception as ex:
                res = [{'error': str(ex)}]
            items = []
            for r in res:
                if 'error' in r:
                    items.append(r)
                    continue
                bbox, text, conf = r[0], r[1], r[2]
                items.append({'text': text, 'conf': round(float(conf), 3),
                              'bbox': [[int(x), int(y)] for x, y in bbox]})
                all_hits.append({'view': vn, 'text': text, 'conf': round(float(conf), 3)})
            rec['views'][vn] = items
        rec['n_hits'] = len(all_hits)
        rec['hits'] = all_hits
        out.append(rec)
        print(f"{os.path.basename(p):38} hits={len(all_hits)}  "
              f"{[h['text'] for h in all_hits][:12]}", flush=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        # easyocr 的检测器会按图像尺寸累积缓存，长批次必被 OOM kill（exit 137）。
        # 每张图后显式释放引用 + gc，才能跑满 100 张。
        # 注意：img/arr 在 with 块退出时已被回收，不能 del，直接清 views 即可。
        views.clear()
        gc.collect()
    print('done ->', args.out)


if __name__ == '__main__':
    main()
