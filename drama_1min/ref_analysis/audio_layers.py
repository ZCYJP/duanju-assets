"""参考片音频分层分析：判断 BGM 音乐床、环境音床、对白、静音占比、动态范围"""
import numpy as np
import subprocess
import json
import os

SR = 16000
WAV = "/tmp/refan/ref16k.wav"


def decode(path, sr=SR):
    cmd = ["/usr/local/bin/ffmpeg", "-v", "error", "-i", path, "-vn",
           "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"]
    r = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


def db(x):
    x = float(np.max(np.abs(x)))
    return 20 * np.log10(x) if x > 1e-12 else -120.0


def rms_env(x, sr=SR, win=0.02):
    w = int(sr * win)
    n = len(x) // w
    if n == 0:
        return np.array([db(x)]), w
    e = x[: n * w].reshape(n, w)
    return 20 * np.log10(np.sqrt((e ** 2).mean(axis=1)) + 1e-12), w


def band_energy(x, sr=SR):
    """16 组频带能量占比"""
    bands = [(0, 60), (60, 150), (150, 300), (300, 600), (600, 1000), (1000, 1500),
             (1500, 2200), (2200, 3000), (3000, 4000), (4000, 5000), (5000, 6500),
             (6500, 8000), (8000, 10000), (10000, 12000), (12000, 14000), (14000, 16000)]
    w = 1024
    hop = 512
    frames = []
    for i in range(0, len(x) - w, hop):
        seg = x[i:i + w] * np.hanning(w)
        frames.append(np.abs(np.fft.rfft(seg)) ** 2)
    if not frames:
        return [0.0] * len(bands), []
    S = np.array(frames)
    freqs = np.fft.rfftfreq(w, 1.0 / sr)
    out = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        out.append(float(S[:, m].sum() / max(S.sum(), 1e-12)))
    return out, S


def spectral_flatness(x, sr=SR):
    w = 1024
    hop = 512
    vals = []
    for i in range(0, len(x) - w, hop, hop * 5):
        seg = x[i:i + w] * np.hanning(w)
        P = np.abs(np.fft.rfft(seg)) ** 2 + 1e-12
        vals.append(float(np.exp(np.mean(np.log(P))) / np.mean(P)))
    return float(np.median(vals)) if vals else 0.0


def main():
    if not os.path.exists(WAV):
        print("WAV not ready:", WAV)
        return
    x = decode(WAV)
    dur = len(x) / SR
    print("=== 参考片音频分层分析 ===")
    print("时长 %.1fs  峰值 %.2f dBFS  RMS %.2f dBFS" % (dur, db(x), db(np.sqrt((x ** 2).mean()))))

    env, w = rms_env(x)
    # 静音阈值：比峰值低 45dB
    thr = env.max() - 45
    sil = env < thr
    print("\n[静音/底噪]")
    print("  环境包络 max %.2f dB  min %.2f dB  P5 %.2f dB  P10 %.2f dB  P50 %.2f dB" % (
        env.max(), env.min(), np.percentile(env, 5), np.percentile(env, 10), np.percentile(env, 50)))
    print("  静音占比(<max-45dB) %.2f%%" % (sil.mean() * 100))
    # 真正的数字静音（-60dBFS 以下绝对阈值）
    print("  绝对静音(<-60dBFS) 占比 %.2f%%" % ((env < -60).mean() * 100))
    print("  低分位噪声底 %.2f dB" % np.percentile(env, 10))

    print("\n[频带能量占比]")
    bands = [(0, 60), (60, 150), (150, 300), (300, 600), (600, 1000), (1000, 1500),
             (1500, 2200), (2200, 3000), (3000, 4000), (4000, 5000), (5000, 6500),
             (6500, 8000), (8000, 10000), (10000, 12000), (12000, 14000), (14000, 16000)]
    be, S = band_energy(x)
    for (lo, hi), v in zip(bands, be):
        print("  %5d-%5dHz : %6.2f%%  %s" % (lo, hi, v * 100, "#" * int(v * 200)))

    print("\n[低频/音乐床判定]")
    # 全片低频持续能量：低频段每帧能量的时间波动
    w2 = 1024
    freqs = np.fft.rfftfreq(w2, 1.0 / SR)
    low_m = freqs < 200
    mid_m = (freqs >= 200) & (freqs < 4000)
    fr = S[:, low_m].sum(axis=1)
    fm = S[:, mid_m].sum(axis=1)
    if len(fr) > 10:
        # 低频能量在"安静段"是否依然存在 → BGM 床
        q = np.percentile(fr, 10)
        quiet = fr <= np.percentile(fr, 30)
        print("  低频(<200Hz)总能量占比 %.2f%%" % (be[0] + be[1] + be[2]))
        print("  低频能量 10分位 %.3e / 中位 %.3e / 90分位 %.3e" % (
            q, np.median(fr), np.percentile(fr, 90)))
        print("  最安静30%%时段中 低频能量中位占比 %.2f%%（>5%% 说明有贯穿音乐床）" % (
            np.median(fr[quiet]) / max(np.median(fr), 1e-12) * 100))
        # 低频节拍周期性（自相关）→ 音乐鼓点
        d = fr - fr.mean()
        ac = np.correlate(d, d, mode="full")[len(d) - 1:]
        ac /= ac[0]
        lo_lag = int(SR / 512 / 2.0)   # 2Hz
        hi_lag = int(SR / 512 / 4.0)   # 4Hz
        if hi_lag > lo_lag:
            seg = ac[lo_lag:hi_lag]
            k = int(np.argmax(seg)) + lo_lag
            print("  低频节拍自相关峰: lag=%d (%.2f Hz) r=%.3f" % (
                k, SR / 512.0 / k, seg.max()))

    print("\n[动态/压缩]")
    print("  谱平坦度(中位) %.4f  (>0.1 宽带噪声)" % spectral_flatness(x))
    p95 = np.percentile(env, 95)
    p10 = np.percentile(env, 10)
    print("  包络 P95-P10 = %.2f dB (短剧旁白压缩通常 15~25dB)" % (p95 - p10))
    print("  包络 P99-P50 = %.2f dB" % (np.percentile(env, 99) - np.percentile(env, 50)))

    # 立体声宽度：检查原始文件左右声道相关性（Opus 立体声）
    cmd = ["/usr/local/bin/ffmpeg", "-v", "error", "-i", "/tmp/ref.mp4", "-vn",
           "-f", "f32le", "-ac", "2", "-ar", "48000", "-"]
    r = subprocess.run(cmd, capture_output=True)
    st = np.frombuffer(r.stdout, dtype=np.float32)
    st = st.reshape(-1, 2)
    if len(st) > 1000:
        L, R = st[:, 0], st[:, 1]
        c = float(np.corrcoef(L, R)[0, 1])
        side = (L - R)
        print("\n[立体声]")
        print("  L/R 相关系数 %.4f (>0.95 = 单声道/几乎居中)" % c)
        print("  Side 能量占比 %.2f%% (=立体声宽度)" % (
            (np.mean(side ** 2) / max(np.mean(L ** 2), 1e-12)) * 100))
        print("  L 峰值 %.2f dBFS  R 峰值 %.2f dBFS" % (db(L), db(R)))

    json.dump({"dur": dur, "peak_db": db(x), "rms_db": db(np.sqrt((x ** 2).mean())),
               "sil_pct": float(sil.mean() * 100), "bands": be,
               "flatness": spectral_flatness(x),
               "p95_p10": float(p95 - p10)},
              open("/tmp/refan/audio_layers.json", "w"), indent=2)
    print("\nwrote /tmp/refan/audio_layers.json")


main()
