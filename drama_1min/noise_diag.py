#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
杂音定位：分层拆解，找噪声到底来自哪一层。
层级：H3 原生音轨 -> TTS 配音 -> 混合后 mix_raw -> 归一后 mix_norm -> 成片
"""
import os, sys, glob, subprocess, wave
import numpy as np

SR = 44100
BASE = "/ssd2/duanju/drama_1min"


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate(); ch = w.getnchannels(); n = w.getnframes()
        raw = w.readframes(n)
    d = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        d = d.reshape(-1, ch).mean(axis=1)
    return d, sr


def rms_db(x):
    if x.size == 0: return -120.0
    return 20*np.log10(max(float(np.sqrt(np.mean(x**2))), 1e-9))


def band_energy(x, sr):
    """频带能量占比（判断嘶声/嗡声/低频隆隆声）"""
    if x.size < 2048: return {}
    X = np.abs(np.fft.rfft(x[:min(len(x), 1<<19)]))
    f = np.fft.rfftfreq(min(len(x), 1<<19), 1/sr)
    total = (X**2).sum() + 1e-12
    out = {}
    for name, lo, hi in [("sub_0_60",0,60),("low_60_250",60,250),("mid_250_2k",250,2000),
                         ("hi_2k_8k",2000,8000),("hiss_8k_16k",8000,16000),("air_16k_22k",16000,22050)]:
        m = (f>=lo)&(f<hi)
        out[name] = float((X[m]**2).sum()/total)
    return out


def spectral_flatness(x, sr):
    """谱平坦度：>0.1 说明有宽带噪声/嘶声"""
    if x.size < 4096: return 0.0
    X = np.abs(np.fft.rfft(x[:min(len(x), 1<<18)])) + 1e-10
    return float(np.exp(np.mean(np.log(X))) / np.mean(X))


def silence_floor(x, sr, win_ms=50):
    """静音段底噪水平"""
    w = int(sr*win_ms/1000)
    n = len(x)//w
    if n == 0: return -120.0
    env = np.array([rms_db(x[i*w:(i+1)*w]) for i in range(n)])
    # 取最低 10% 的窗作为底噪
    k = max(1, n//10)
    return float(np.sort(env)[:k].mean())


def tonal(x, sr):
    """检测单频嗡声：找能量最大的窄带峰"""
    if x.size < 4096: return (0, 0.0)
    X = np.abs(np.fft.rfft(x[:min(len(x), 1<<19)]))
    f = np.fft.rfftfreq(min(len(x), 1<<19), 1/sr)
    X[f<20] = 0
    k = int(X.max()*0.5)
    idx = np.where(X > k)[0]
    if len(idx)==0: return (0, 0.0)
    # 找最集中的峰
    best_f, best_n = 0, 0
    for i in idx:
        c = int(np.sum((f > max(0,f[i]-8)) & (f < f[i]+8)))
        if c > best_n:
            best_n, best_f = c, float(f[i])
    return (best_f, float(best_n))


print("="*72)
print("A. H3 原生音轨逐镜诊断（混音前）")
print("="*72)
print(f"{'镜头':<22}{'RMS':>8}{'底噪':>9}{'平坦度':>9}{'嘶声%':>9}{'隆隆%':>9}{'主峰Hz':>10}")
for p in sorted(glob.glob(f"{BASE}/motion_h3_fln/*.mp4")):
    n = os.path.basename(p)[:-4]
    w = f"/tmp/noise/{n}.wav"
    if not os.path.exists(w): continue
    x, sr = read_wav(w)
    b = band_energy(x, sr)
    ff, _ = tonal(x, sr)
    print(f"{n:<22}{rms_db(x):>8.1f}{silence_floor(x,sr):>9.1f}"
          f"{spectral_flatness(x,sr):>9.4f}{b.get('hiss_8k_16k',0)*100:>9.2f}"
          f"{b.get('sub_0_60',0)*100:>9.2f}{ff:>10.1f}")

print()
print("="*72)
print("B. TTS 配音逐句诊断")
print("="*72)
print(f"{'文件':<34}{'RMS':>8}{'底噪':>9}{'平坦度':>9}{'嘶声%':>9}{'主峰Hz':>10}")
for p in sorted(glob.glob(f"{BASE}/voice/*.wav")):
    x, sr = read_wav(p)
    b = band_energy(x, sr)
    ff, _ = tonal(x, sr)
    print(f"{os.path.basename(p)[:-4]:<34}{rms_db(x):>8.1f}{silence_floor(x,sr):>9.1f}"
          f"{spectral_flatness(x,sr):>9.4f}{b.get('hiss_8k_16k',0)*100:>9.2f}{ff:>10.1f}")

print()
print("="*72)
print("C. 混合链各层对比")
print("="*72)
layers = [
    ("H3原生(12镜拼接参考)", f"{BASE}/compose_h3_work/mix_raw.wav"),
    ("mix_raw 混合后", f"{BASE}/compose_h3_work/mix_raw.wav"),
    ("mix_norm 响度归一后", f"{BASE}/compose_h3_work/mix_norm.wav"),
]
for label, p in layers:
    if not os.path.exists(p):
        print(f"{label:<24} 缺失")
        continue
    x, sr = read_wav(p)
    b = band_energy(x, sr)
    ff, _ = tonal(x, sr)
    print(f"{label:<24} RMS={rms_db(x):>7.1f} 底噪={silence_floor(x,sr):>7.1f} "
          f"平坦度={spectral_flatness(x,sr):.4f} 嘶声={b.get('hiss_8k_16k',0)*100:.2f}% "
          f"隆隆={b.get('sub_0_60',0)*100:.2f}% 主峰={ff:.1f}Hz")

print()
print("="*72)
print("D. 成片音轨")
print("="*72)
out = "/tmp/noise/final.wav"
subprocess.run(["/ssd2/miniconda3/envs/videogen/bin/ffmpeg","-y","-v","error",
                "-i",f"{BASE}/final/drama_1min_h3.mp4","-vn","-ac","1","-ar","44100",
                "-c:a","pcm_s16le",out], check=True)
x, sr = read_wav(out)
b = band_energy(x, sr)
ff, _ = tonal(x, sr)
print(f"成片音轨            RMS={rms_db(x):>7.1f} 底噪={silence_floor(x,sr):>7.1f} "
      f"平坦度={spectral_flatness(x,sr):.4f} 嘶声={b.get('hiss_8k_16k',0)*100:.2f}% "
      f"隆隆={b.get('sub_0_60',0)*100:.2f}% 主峰={ff:.1f}Hz")

# 分镜分段的底噪（判断是环境音铺底还是真噪声）
print("\n=== 成片逐 5s 分段底噪/峰值 ===")
w = int(sr*0.5)
for i in range(0, len(x)//w):
    seg = x[i*w:(i+1)*w]
    if len(seg) < 100: continue
    b2 = band_energy(seg, sr)
    print(f"  {i*0.5:>5.1f}s  RMS={rms_db(seg):>7.1f}  底噪={silence_floor(seg,sr):>7.1f}  "
          f"嘶声={b2.get('hiss_8k_16k',0)*100:>5.2f}%")
