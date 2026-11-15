#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《决裂》配音独立质检 — 只读 voice/*.wav，不重新生成。

F0 算法：逐帧（40ms窗/20ms跳）RMS 门限过滤 + 自相关取峰 + 中位数，
与 /ssd2/ysyj/tts_qwen_remote.py 的 rough_f0 同源（已验证可靠）。
（原 gen_voice.py 的整段 FFT 自相关在短句/低能量段会返回 0，导致漏检与误熔断。）

门禁：
  女声：中位 F0 > 175Hz；< 150Hz 熔断；150~175Hz 告警
  男声：70 <= F0 <= 220
  通用：时长 0.4~9.0s、峰值 0.05~0.98、语速 2.5~9.0 字/秒

用法：qc_voice.py [--json 输出路径]
"""
import os, sys, json, wave
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shots_1min import SHOTS, NAME2CHAR

BASE = "/ssd2/duanju/drama_1min"
VOICE_DIR = f"{BASE}/voice"

FEMALE = {"F01", "F03", "F08"}
MALE = {"M01", "M03"}

# 角色名 -> 台词（从 shots 取），用于语速检查
TEXT_MAP = {}
for _s in SHOTS:
    for _sp, _tx in _s["dialogue"]:
        TEXT_MAP[f"{_s['name']}_{_sp}"] = _tx


def read_wav(path):
    with wave.open(path, "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


def rough_f0(x, sr):
    frame, hop = int(sr * 0.04), int(sr * 0.02)
    vals = []
    for st in range(0, len(x) - frame, hop):
        seg = x[st:st + frame].astype(np.float64)
        if np.sqrt(np.mean(seg ** 2)) < 0.012:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, mode="full")[frame - 1:]
        lo, hi = int(sr / 400), int(sr / 70)
        if hi >= len(ac):
            continue
        vals.append(sr / (lo + int(np.argmax(ac[lo:hi]))))
    return float(np.median(vals)) if vals else 0.0


def qc(path):
    x, sr = read_wav(path)
    base = os.path.basename(path)[:-4]
    # 文件名形如 {shot}_{角色}，角色在最后一段（split("_") 全部切分取末段）
    parts = base.split("_")
    speaker = parts[-1]
    cid = NAME2CHAR.get(speaker)
    gender = "female" if cid in FEMALE else ("male" if cid in MALE else "unknown")
    text = TEXT_MAP.get(base, "")
    dur = len(x) / sr
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    f0 = rough_f0(x, sr)
    cps = len(text) / max(dur, 0.01) if text else 0.0

    issues = []
    if dur < 0.4:
        issues.append(f"过短{dur:.2f}s")
    if dur > 9.0:
        issues.append(f"过长{dur:.2f}s")
    if peak > 0.98:
        issues.append(f"削波peak={peak:.3f}")
    if peak < 0.05:
        issues.append(f"音量过低peak={peak:.3f}")
    if text and not (2.5 <= cps <= 9.0):
        issues.append(f"语速异常{cps:.1f}字/秒")
    if f0 <= 0:
        issues.append("F0无法测量")
    elif gender == "female":
        if f0 < 150:
            issues.append(f"女声F0={f0:.0f}Hz<150 熔断")
        elif f0 < 175:
            issues.append(f"女声F0={f0:.0f}Hz<175")
    elif gender == "male":
        if not (70 <= f0 <= 220):
            issues.append(f"男声F0={f0:.0f}Hz越界")

    return {"file": base, "char": cid, "speaker": speaker, "gender": gender, "text": text,
            "dur": round(dur, 3), "peak": round(peak, 4), "f0": round(f0, 1),
            "cps": round(cps, 2), "issues": issues}


def main():
    out_json = None
    if "--json" in sys.argv:
        i = sys.argv.index("--json")
        out_json = sys.argv[i + 1] if i + 1 < len(sys.argv) else f"{VOICE_DIR}/qc_report.json"

    files = sorted(f for f in os.listdir(VOICE_DIR) if f.endswith(".wav"))
    rows = []
    for f in files:
        r = qc(os.path.join(VOICE_DIR, f))
        rows.append(r)
        flag = "FAIL" if r["issues"] else "OK  "
        print(f"[{flag}] {r['file']:<28} {r['gender']:<7} {r['dur']:>5.2f}s "
              f"f0={r['f0']:>6.1f}Hz peak={r['peak']:.3f} cps={r['cps']:>4.1f} "
              f"{r['issues'] if r['issues'] else ''}")

    bad = [r for r in rows if r["issues"]]
    print(f"\n总计 {len(rows)} 句，问题 {len(bad)} 句")
    for r in bad:
        print("  FAIL", r["file"], r["issues"])

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print("报告已写:", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
