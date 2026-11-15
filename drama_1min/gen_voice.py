#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
《决裂》1分钟短剧 — 真音配音（Qwen3-TTS 声纹克隆）

5 个角色 -> 4 个原创声纹参考音（不混用同一性别不同角色的音高，防止撞声）：
  M01 陆沉   -> 02_学者男声_t1 (F0 102.8Hz, 42岁男)
  M03 裴九渊 -> 01_诵经男声_t1 (F0 107.4Hz, 55岁男, 更沉更阴鸷)
  F01 苏清晚 -> 04_清朗女声_t1 (27岁女, 清亮)
  F03 苏蔓   -> 03_温润女声_t1 (32岁女, 温婉知性)
  F08 宋佳宁 -> 03_温润女声_t1 (32岁女)

F0 门禁：女声中位F0 必须 >175Hz，<150Hz 立即熔断（防TTS变异男声）
每个角色取 3 个 seed，选 F0/峰值/漂移全合格的最佳候选。

输出：voice/{shot}_{speaker}.wav  (44.1k mono 16bit) + voice/f0_report.json
用法：gen_voice.py
"""
import os, sys, json, wave, traceback
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from math import gcd


def _patch_transformers_check_model_inputs():
    """
    兼容补丁：qwen-tts 0.1.1 使用 transformers>=5.21 的 `@check_model_inputs()`（无参）写法，
    而本机 transformers 5.20.2 的 `check_model_inputs(func)` 是单参旧签名，直接 import 会报
    TypeError: check_model_inputs() missing 1 required positional argument: 'func'。
    这里在 import qwen_tts 之前把 generic.check_model_inputs 换成同时支持两种写法的版本：
      - @check_model_inputs      (旧式，传 func)
      - @check_model_inputs()    (新式，无参 -> 返回 merge_with_config_defaults 装饰器)
    不降级/不升级 transformers，避免破坏同环境下的 ComfyUI。
    """
    try:
        import transformers
        import transformers.utils.generic as _g
    except Exception:
        return
    # transformers 4.57.x 的 check_model_inputs 是单参旧签名（与 qwen_tts 匹配），无需补丁
    if not hasattr(_g, "merge_with_config_defaults"):
        return
    if getattr(_g, "_wb_patched_check_model_inputs", False):
        return
    _orig = _g.check_model_inputs

    def _compat(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return _orig(args[0])           # 旧式 @check_model_inputs
        # 新式 @check_model_inputs()：返回真正的 config-defaults 装饰器
        return getattr(_g, "merge_with_config_defaults", _orig)

    _compat._wb_patched_check_model_inputs = True
    _g.check_model_inputs = _compat
    try:
        import transformers
        transformers.utils.generic.check_model_inputs = _compat
    except Exception:
        pass
    print("[compat] transformers.check_model_inputs 兼容补丁已生效", flush=True)


_patch_transformers_check_model_inputs()


def _patch_qwen_tts_config():
    """
    兼容补丁2：transformers 5.20.2 不会给 PretrainedConfig 子类自动注入标准属性 `pad_token_id`，
    而 qwen_tts 0.1.1 的 Qwen3TTSTalkerModel.__init__ 直接读 config.pad_token_id -> AttributeError。
    模型 config.json 里 talker_config 实际只有 codec_pad_id(2148)/codec_eos_token_id(2150)，
    历史成功日志（ysyj/qwen_voices/rebuild.log）显示当时生效值为 eos_token_id=2150。
    这里在配置类实例化后补齐 pad_token_id = codec_eos_token_id。
    """
    try:
        from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSTalkerConfig
    except Exception as e:
        print("[compat] 跳过 config 补丁（导入失败）:", e, flush=True)
        return
    if getattr(Qwen3TTSTalkerConfig, "_wb_patched_pad_token_id", False):
        return
    # 4.57.x 的 PretrainedConfig 会自动注入 pad_token_id，无需补丁
    import transformers
    if not hasattr(transformers, "__version__") or not transformers.__version__.startswith("5."):
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


_patch_qwen_tts_config()


def _patch_rope_default():
    """
    兼容补丁3：transformers 5.20.2 把默认 RoPE 从 ROPE_INIT_FUNCTIONS 注册表中移除
    （原 'default' 键并入 'proportional'），而 qwen_tts 0.1.1 的
    Qwen3TTSTalkerRotaryEmbedding 在 rope_scaling 为 None 时会取 ROPE_INIT_FUNCTIONS['default']
    -> KeyError: 'default'。
    5.20.2 中 proportional 在 partial_rotary_factor=1 时与旧 default 行为等价（attention_factor=1.0），
    这里把 'default' 重新注册到 proportional 实现上。
    """
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    except Exception as e:
        print("[compat] 跳过 rope 补丁:", e, flush=True)
        return
    if "default" in ROPE_INIT_FUNCTIONS:
        return  # 4.57.x 保留了 default，无需补丁
    if "proportional" in ROPE_INIT_FUNCTIONS:
        ROPE_INIT_FUNCTIONS["default"] = ROPE_INIT_FUNCTIONS["proportional"]
        print("[compat] ROPE_INIT_FUNCTIONS['default'] 已补注册到 proportional", flush=True)


_patch_rope_default()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shots_1min import SHOTS, NAME2CHAR, VOICE_REF

BASE = "/ssd2/duanju/drama_1min"
VOICE_DIR = f"{BASE}/voice"
RAW_DIR = f"{VOICE_DIR}/raw"
os.makedirs(RAW_DIR, exist_ok=True)

MODEL_DIR = "/ssd2/models/Qwen3-TTS-12Hz-1.7B-Base"
REF_TEXT = "一人之身，一國之象。把身体当成一个国家来看：精神是君主，气是百姓，爱气才能全身。"
SR, CH, SW = 44100, 1, 2
SEEDS = [42, 43, 44]

# 角色性别（用于F0门禁）
FEMALE = {"F01", "F03", "F08"}
MALE = {"M01", "M03"}


def f0_median(w, sr):
    """用 autocorr 估算中位基频（无需额外依赖）"""
    if len(w) < sr // 2:
        return 0.0
    x = w[:sr * 6] if len(w) > sr * 6 else w
    x = x - x.mean()
    if np.max(np.abs(x)) < 1e-6:
        return 0.0
    # 自相关
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= (ac[0] + 1e-12)
    lo, hi = int(sr / 400), int(sr / 60)     # 60-400Hz
    if hi >= n:
        hi = n - 1
    seg = ac[lo:hi]
    if len(seg) == 0:
        return 0.0
    # 取若干峰值的中位
    idx = []
    for i in range(1, len(seg) - 1):
        if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.3:
            idx.append(i)
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
    gpu = os.environ.get("TTS_GPU", "0")
    print(f"[1/4] 加载模型... (cuda:{gpu})", flush=True)
    model = Qwen3TTSModel.from_pretrained(
        MODEL_DIR, device_map=f"cuda:{gpu}", dtype=torch.bfloat16, attn_implementation="sdpa")

    # 建立每个参考音的声纹 profile
    used_refs = sorted(set(v[0] for v in VOICE_REF.values()))
    print("[2/4] 建立声纹 profile:", used_refs, flush=True)
    profiles = {}
    for rp in used_refs:
        item = model.create_voice_clone_prompt(rp, REF_TEXT)[0]
        profiles[rp] = item
        print("   profile OK:", os.path.basename(rp), flush=True)

    # 收集台词任务
    jobs = []
    for s in SHOTS:
        for sp, tx in s["dialogue"]:
            cid = NAME2CHAR[sp]
            rp = VOICE_REF[cid][0]
            jobs.append((s["name"], sp, cid, rp, tx))

    print(f"[3/4] 生成 {len(jobs)} 句配音...", flush=True)
    report = []
    for name, sp, cid, rp, tx in jobs:
        raw_path = f"{RAW_DIR}/{name}_{sp}.wav"
        out_path = f"{VOICE_DIR}/{name}_{sp}.wav"
        gender = "female" if cid in FEMALE else "male"
        # 缓存命中
        if os.path.exists(raw_path):
            w, sr = sf.read(raw_path, dtype="float32")
            r = qc(w, sr, gender)
            if not r["issues"]:
                write_wav(out_path, to_pcm(w, sr))
                r.update({"shot": name, "speaker": sp, "char": cid, "text": tx, "seed": "cached"})
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
                r.update({"shot": name, "speaker": sp, "char": cid, "text": tx})
                report.append(r)
                print(f"   [OK] {name}_{sp} seed={seed} {r['dur']}s f0={r['f0']}Hz", flush=True)
                break
            print(f"   [retry] {name}_{sp} seed={seed} {r['issues']}", flush=True)
            if best is None or len(r["issues"]) < len(best[2]["issues"]):
                best = (w, sr, r)
        if best and not os.path.exists(out_path):
            w, sr, r = best
            write_wav(out_path, to_pcm(w, sr))
            r.update({"shot": name, "speaker": sp, "char": cid, "text": tx})
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
