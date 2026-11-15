#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《零点四十七单》— 真音配音（Qwen3-TTS-Base 声纹克隆）

核心规则：一角色一参考音，绝不共用。
参考音 = 该角色在 /ssd2/duanju/voices/ 下的专属基线音色（VoiceDesign 生成）
克隆实测：M52 F0 偏差 0.0%，F25 偏差 1.4%。

F0 门禁：女声中位 F0 必须 >175Hz（<150 立即熔断），男声须落在 70-220Hz。
每句取 3 个 seed，选 F0/峰值/时长全合格的最佳候选。

输出：voice/{shot}_{speaker}.wav (44.1k mono 16bit) + voice/f0_report.json
用法：gen_voice.py [shot_name ...]   不带参数 = 全部
"""
import os
import sys
import json
import wave
import traceback
from math import gcd

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly


def _patch_transformers_check_model_inputs():
    try:
        import transformers
        import transformers.utils.generic as _g
    except Exception:
        return
    if not hasattr(_g, "merge_with_config_defaults"):
        return
    if getattr(_g, "_wb_patched_check_model_inputs", False):
        return
    _orig = _g.check_model_inputs

    def _compat(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return _orig(args[0])
        return getattr(_g, "merge_with_config_defaults", _orig)

    _compat._wb_patched_check_model_inputs = True
    _g.check_model_inputs = _compat
    try:
        import transformers
        transformers.utils.check_model_inputs = _compat
    except Exception:
        pass
    print("[compat] transformers.check_model_inputs 兼容补丁已生效", flush=True)


def _patch_qwen_tts_config():
    try:
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
    except Exception:
        print("[compat] 跳过 qwen config 补丁（导入失败）", flush=True)
        return
    import transformers
    if not transformers.__version__.startswith("5."):
        return
    if getattr(Qwen3TTSTalkerConfig, "_wb_patched_pad_token_id", False):
        return
    _orig_init = Qwen3TTSTalkerConfig.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        if getattr(self, "pad_token_id", None) is None:
            for attr in ("codec_eos_token_id", "codec_pad_id"):
                v = getattr(self, attr, None)
                if v is not None:
                    self.pad_token_id = v
                    break

    Qwen3TTSTalkerConfig.__init__ = _patched_init
    Qwen3TTSTalkerConfig._wb_patched_pad_token_id = True
    print("[compat] Qwen3TTSTalkerConfig.pad_token_id 补丁已生效", flush=True)


def _patch_rope_default():
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except Exception:
        return
    if "default" in ROPE_INIT_FUNCTIONS:
        return
    if "proportional" in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = ROPE_INIT_FUNCTIONS["proportional"]
        print("[compat] ROPE_INIT_FUNCTIONS['default'] 已补注册", flush=True)


_patch_transformers_check_model_inputs()
_patch_qwen_tts_config()
_patch_rope_default()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from story import SHOTS, NAME2CHAR, VOICE_REF, FEMALE, BASE

VOICE_DIR = f"{BASE}/voice"
RAW_DIR = f"{VOICE_DIR}/raw"
os.makedirs(RAW_DIR, exist_ok=True)

MODEL_DIR = "/ssd2/models/Qwen3-TTS-12Hz-1.7B-Base"
# 参考音朗读文本（duanju/voices/samples* 生成时的固定文本，克隆必须对齐）
REF_TEXT = "这条路我走了很久，今天，终于轮到我来定规矩。"
SR, CH, SW = 44100, 1, 2
SEEDS = [42, 43, 44]


def f0_median(w, sr):
    if len(w) < sr // 2:
        return 0.0
    x = w[:sr * 8]
    x = x - x.mean()
    if np.max(np.abs(x)) < 1e-6:
        return 0.0
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= (ac[0] + 1e-12)
    lo, hi = int(sr / 400), int(sr / 60)
    if hi >= n:
        hi = n - 1
    seg = ac[lo:hi]
    if len(seg) == 0:
        return 0.0
    idx = [i for i in range(1, len(seg) - 1)
           if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.3]
    if not idx:
        return 0.0
    best = idx[:12]
    return float(np.median([sr / (i + lo) for i in best]))


def qc(w, sr, gender):
    issues = []
    dur = len(w) / sr
    peak = float(np.max(np.abs(w))) if len(w) else 0.0
    f0 = f0_median(w, sr)
    if dur < 0.4:
        issues.append(f"过短{dur:.2f}s")
    if dur > 9.0:
        issues.append(f"过长{dur:.2f}s")
    if peak > 0.98:
        issues.append(f"削波peak={peak:.3f}")
    if peak < 0.05:
        issues.append(f"音量过低peak={peak:.3f}")
    if gender == "female":
        if f0 and f0 < 150:
            issues.append(f"女声F0={f0:.0f}Hz<150 熔断")
        elif f0 and f0 < 175:
            issues.append(f"女声F0={f0:.0f}Hz<175")
    if gender == "male":
        if f0 and (f0 < 70 or f0 > 220):
            issues.append(f"男声F0={f0:.0f}Hz异常")
    return {"issues": issues, "dur": round(dur, 3), "peak": round(peak, 4),
            "f0": round(f0, 1), "gender": gender}


def to_pcm(w, sr, gain=0.92):
    if sr != SR:
        g = gcd(SR, sr)
        w = resample_poly(w, SR // g, sr // g).astype(np.float32)
    w = np.clip(w * gain, -0.99, 0.99)
    return (w * 32767).astype(np.int16).tobytes()


def write_wav(path, pcm_bytes):
    with wave.open(path, "wb") as w:
        w.setnchannels(CH)
        w.setsampwidth(SW)
        w.setframerate(SR)
        w.writeframes(pcm_bytes)


def main():
    from qwen_tts import Qwen3TTSModel
    gpu = os.environ.get("TTS_GPU", "1")
    only = set(sys.argv[1:])
    shots = [s for s in SHOTS if not only or s["name"] in only]

    jobs = []
    for s in shots:
        for sp, tx in s["dialogue"]:
            cid = NAME2CHAR[sp]
            jobs.append((s["name"], sp, cid, VOICE_REF[cid], tx))

    if not jobs:
        print("没有台词任务")
        return 0

    print(f"[1/4] 加载模型... (cuda:{gpu})", flush=True)
    model = Qwen3TTSModel.from_pretrained(
        MODEL_DIR, device_map=f"cuda:{gpu}", dtype=torch.bfloat16,
        attn_implementation="sdpa")

    used_refs = sorted(set(j[3] for j in jobs))
    print("[2/4] 建立声纹 profile:", [os.path.basename(r) for r in used_refs], flush=True)
    profiles = {}
    for rp in used_refs:
        profiles[rp] = model.create_voice_clone_prompt(rp, REF_TEXT)[0]
        print("   profile OK:", os.path.basename(rp), flush=True)

    print(f"[3/4] 生成 {len(jobs)} 句配音...", flush=True)
    report = []
    for name, sp, cid, rp, tx in jobs:
        raw_path = f"{RAW_DIR}/{name}_{sp}.wav"
        out_path = f"{VOICE_DIR}/{name}_{sp}.wav"
        gender = "female" if cid in FEMALE else "male"
        if os.path.exists(raw_path):
            w, sr = sf.read(raw_path, dtype="float32")
            r = qc(w, sr, gender)
            if not r["issues"]:
                write_wav(out_path, to_pcm(w, sr))
                r.update({"shot": name, "speaker": sp, "char": cid, "text": tx,
                          "ref": rp, "seed": "cached"})
                report.append(r)
                print(f"   [cached] {name}_{sp} {r['dur']}s f0={r['f0']}", flush=True)
                continue
        best = None
        for seed in SEEDS:
            torch.manual_seed(seed)
            wavs, sr = model.generate_voice_clone(tx, "Chinese", voice_clone_prompt=[profiles[rp]])
            w = np.asarray(wavs[0], dtype=np.float32)
            r = qc(w, sr, gender)
            r["seed"] = seed
            sf.write(raw_path, w, sr)
            if not r["issues"]:
                write_wav(out_path, to_pcm(w, sr))
                r.update({"shot": name, "speaker": sp, "char": cid, "text": tx, "ref": rp})
                report.append(r)
                print(f"   [OK] {name}_{sp} seed={seed} {r['dur']}s f0={r['f0']}Hz", flush=True)
                break
            print(f"   [retry] {name}_{sp} seed={seed} {r['issues']}", flush=True)
            if best is None or len(r["issues"]) < len(best[2]["issues"]):
                best = (w, sr, r)
        if best and not os.path.exists(out_path):
            w, sr, r = best
            write_wav(out_path, to_pcm(w, sr))
            r.update({"shot": name, "speaker": sp, "char": cid, "text": tx, "ref": rp})
            r["issues"].append("全部seed未全过，采用最佳候选")
            report.append(r)
            print(f"   [BEST] {name}_{sp} {r['dur']}s f0={r['f0']}Hz {r['issues']}", flush=True)

    print("[4/4] 写 F0 报告...", flush=True)
    with open(f"{VOICE_DIR}/f0_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    bad = [r for r in report if r["issues"]]
    print(f"\n完成 {len(report)} 句，问题 {len(bad)} 句")
    for r in bad:
        print("  ", r["shot"], r["speaker"], r["issues"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
