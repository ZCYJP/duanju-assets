#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定位 probe 出图的 'can't concat str to bytes' 错误来源。"""
import os
import sys
import traceback

sys.path.insert(0, '/ssd2/short_drama/dh_scripts')
sys.path.insert(0, '/ssd2/duanju/scripts')
os.environ['COMFY_URL'] = 'http://127.0.0.1:8188'

import comfy  # noqa: E402
import probe_gen  # noqa: E402

print('--- comfy.URL =', comfy.URL, flush=True)

print('--- step1: blank_png ---', flush=True)
try:
    p = probe_gen.blank_png('blank:832x1472')
    print('  ok', p, os.path.getsize(p), flush=True)
except Exception:
    traceback.print_exc()

print('--- step2: PIL 转 JPEG ---', flush=True)
try:
    import tempfile
    import shutil
    from PIL import Image
    tmpdir = tempfile.mkdtemp(prefix='cimg_')
    dst = os.path.join(tmpdir, 'img.jpg')
    with Image.open(p) as im:
        im.convert('RGB').save(dst, quality=96, subsampling=0)
    print('  ok', dst, os.path.getsize(dst), flush=True)
except Exception:
    traceback.print_exc()

print('--- step3: comfy.upload ---', flush=True)
try:
    name = comfy.upload(dst)
    print('  ok uploaded:', name, flush=True)
except Exception:
    traceback.print_exc()

print('--- step4: mimetypes 探测 ---', flush=True)
import mimetypes  # noqa: E402
print('  guess_type(img.jpg) =', mimetypes.guess_type('img.jpg'), flush=True)
print('  guess_type(img.jpg) =', mimetypes.guess_type(os.path.basename(dst)), flush=True)
print('  python', sys.version, flush=True)
