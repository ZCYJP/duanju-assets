#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 dh_scripts/comfy.py 的 download() 加下载重试 + PNG 完整性校验。

背景（2026-11-15）：并发调用 ComfyUI 时，服务端写盘/传输会被截断，
urllib 抛 IncompleteRead，结果落盘的是半截 PNG（后续 PIL 报
"cannot identify image file"）。修复：失败重试 3 次，
且每次落盘后校验 PNG 是否完整（能被 PIL 解码），不完整则丢弃重试。

同时提供 from_comfy_output_dir()：ComfyUI 的 --output-directory 与
下载目录是同一个，产物其实已在服务端落盘，可直接本地取用，
避免不必要的网络下载。
"""
import io
import os
import shutil
import sys

SRC = '/ssd2/short_drama/dh_scripts/comfy.py'


def png_ok(path):
    """校验文件是完整可解码的图片。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


NEW_DOWNLOAD = '''def download(entry, out_dir):
    """Save every output file of a finished job into out_dir; returns the local paths.

    2026-11-15: 加下载重试 + 完整性校验。
    并发场景下服务端写盘/传输会被截断，urllib 抛 IncompleteRead，
    落盘的是半截 PNG（PIL 报 cannot identify image file）。
    """
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for o in entry.get('outputs', {}).values():
        for k in ('images', 'videos', 'gifs'):
            for it in o.get(k, []):
                if it.get('type', 'output') != 'output':
                    continue
                q = urllib.parse.urlencode({'filename': it['filename'], 'subfolder': it.get('subfolder', ''),
                                            'type': 'output'})
                dst = os.path.join(out_dir, it['filename'])
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                got = False
                for attempt in range(4):
                    try:
                        with urllib.request.urlopen(f'{URL}/view?{q}', timeout=600) as r:
                            data = r.read()
                        with open(dst, 'wb') as f:
                            f.write(data)
                        # 完整性校验：PNG 必须能被解码，否则视为截断
                        if _image_ok(dst):
                            got = True
                            break
                        os.remove(dst)
                    except Exception:
                        try:
                            if os.path.exists(dst):
                                os.remove(dst)
                        except OSError:
                            pass
                    time.sleep(1.5 * (attempt + 1))
                if not got:
                    raise IOError(f'download failed or truncated after retries: {it["filename"]}')
                files.append(dst)
    return files


def _image_ok(path):
    """校验文件是完整可解码的图片（PNG 截断时 PIL 会抛异常）。"""
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def from_comfy_output_dir(filename, search_dirs):
    """ComfyUI 的 --output-directory 通常与调用方 out_dir 相同，
    产物其实已在服务端落盘。下载失败时直接从服务端目录取用。

    search_dirs: 候选目录列表（按优先级）。
    返回本地文件路径或 None。
    """
    base = os.path.basename(filename)
    for d in search_dirs:
        if not d:
            continue
        cand = os.path.join(d, filename)
        if os.path.exists(cand) and _image_ok(cand):
            return cand
        # 也可能在子目录里
        sub = os.path.join(d, os.path.dirname(filename))
        if os.path.isdir(sub):
            for f in os.listdir(sub):
                if f == base or f.startswith(base.rsplit('.', 1)[0]):
                    p = os.path.join(sub, f)
                    if _image_ok(p):
                        return p
    return None
'''


def main():
    with open(SRC, encoding='utf-8') as f:
        src = f.read()

    if '_image_ok' in src:
        print('already patched')
        return 0

    start = src.index('def download(entry, out_dir):')
    # 找到 download 函数结束（下一个顶层 def/class）
    rest = src[start:]
    lines = rest.split('\n')
    end = None
    for i, ln in enumerate(lines[1:], start=1):
        if ln.startswith('def ') or ln.startswith('class '):
            end = i
            break
    if end is None:
        end = len(lines)
    old_block = '\n'.join(lines[:end])

    # 旧块后面若有 _download 之类的引用也要一起替换
    new_src = src.replace(old_block, NEW_DOWNLOAD.rstrip('\n'))
    with open(SRC, 'w', encoding='utf-8') as f:
        f.write(new_src)
    print('patched download() in', SRC)

    # 语法检查
    import py_compile
    py_compile.compile(SRC, doraise=True)
    print('syntax OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
