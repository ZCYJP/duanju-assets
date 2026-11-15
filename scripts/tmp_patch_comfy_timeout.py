#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给 comfy.run() 加总超时与断连检测，修死等 bug。

原实现：
    while True:
        try: h = call(f'/history/{pid}')
        except Exception: continue      # ← 异常吞掉，无限重试
        if pid in h: return h[pid]

问题：ComfyUI 实例在任务执行中崩溃/重启后，/history/{pid} 永远查不到该 prompt_id，
而单次 call 的 60s 超时被 except 吞掉 → 进程永久死等，实测挂住 20+ 分钟。

修复：
  1. 连续异常达 N 次 → 判定实例失联，抛 RuntimeError（快速失败）
  2. 总等待超过 max_wait 秒 → 抛 RuntimeError（防止慢任务无限等）
  3. 异常时先探测 /system_stats 确认实例是否还活着，区分「加载中」与「已挂」
"""
import os, shutil, sys

TARGET = "/ssd2/short_drama/dh_scripts/comfy.py"
BAK = TARGET + ".bak_gender"
MARK = "# ==== RUN TIMEOUT GUARD (auto-patched) ===="

OLD = '''def run(graph, poll=3):
    """Queue an API-format graph and wait for it; returns the history entry."""
    pid = call('/prompt', {'prompt': graph, 'client_id': CLIENT})['prompt_id']
    while True:
        time.sleep(poll)
        try:
            h = call(f'/history/{pid}')
        except Exception:          # the probe can time out while a big model loads; the job keeps running
            continue
        if pid in h:
            return h[pid]'''

NEW = '''def run(graph, poll=3, max_wait=900, max_err=8):
    """Queue an API-format graph and wait for it; returns the history entry.

    ==== RUN TIMEOUT GUARD (auto-patched) ====
    原实现对 /history 探测异常无条件 continue，而 ComfyUI 实例在任务执行中
    崩溃/重启后该 prompt_id 永远不会出现 —— 进程会无限死等（实测 20+ 分钟）。
    现在加了总时限与连续异常上限，失败时快速抛错让调用方重试。
    """
    pid = call('/prompt', {'prompt': graph, 'client_id': CLIENT})['prompt_id']
    # ==== RUN TIMEOUT GUARD (auto-patched) ====
    t0 = time.time()
    errs = 0
    while True:
        time.sleep(poll)
        if time.time() - t0 > max_wait:
            raise RuntimeError(
                f'comfy.run timeout after {max_wait}s (prompt_id={pid}); '
                f'queue still running? check ComfyUI instance {URL}')
        try:
            h = call(f'/history/{pid}', timeout=30)
        except Exception as ex:
            errs += 1
            # 探测实例是否还活着：活着说明只是暂时繁忙/加载中，死了说明要快速失败
            try:
                call('/system_stats', timeout=10)
                alive = True
            except Exception:
                alive = False
            if not alive or errs >= max_err:
                raise RuntimeError(
                    f'comfy.run failed: instance unreachable (alive={alive}, '
                    f'consecutive_errors={errs}, elapsed={time.time()-t0:.0f}s, '
                    f'prompt_id={pid}, last_error={str(ex)[:120]})')
            continue
        errs = 0
        if pid in h:
            return h[pid]
    # ==== END RUN TIMEOUT GUARD ===='''


def main():
    if not os.path.exists(BAK):
        shutil.copy2(TARGET, BAK)
        print(f"backup -> {BAK}")
    src = open(TARGET, encoding="utf-8").read()
    if MARK in src:
        print("already patched")
        return
    if OLD not in src:
        print("!! pattern not found")
        sys.exit(1)
    src = src.replace(OLD, NEW, 1)
    open(TARGET, "w", encoding="utf-8").write(src)
    import py_compile
    py_compile.compile(TARGET, doraise=True)
    print(f"patched + syntax OK: {TARGET}")


if __name__ == "__main__":
    main()
