"""精确 BGM 判定：逐 0.5s 切片，区分【真静音】【背景床(低能量非静音)】【语音主导】"""
import numpy as np
import subprocess
import json

SR = 48000
FF = "/usr/local/bin/ffmpeg"


def decode(path):
    cmd = [FF, "-v", "error", "-i", path, "-vn", "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"]
    r = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


x = decode("/tmp/ref.mp4")
dur = len(x) / SR
print("时长 %.1fs" % dur)

NFFT = 2048
HOP = 1024
hop_s = HOP / SR
n = 1 + (len(x) - NFFT) // HOP
idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
frames = x[idx] * np.hanning(NFFT)[None, :]
S = np.abs(np.fft.rfft(frames, axis=1)) ** 2
freqs = np.fft.rfftfreq(NFFT, 1.0 / SR)
e = S.sum(axis=1)
e_db = 10 * np.log10(e + 1e-20)

peak = e_db.max()
# 语音带集中度
sp_band = (freqs >= 300) & (freqs <= 3400)
sp_conc = S[:, sp_band].sum(axis=1) / (e + 1e-20)
lo_band = freqs < 200
lo_e = S[:, lo_band].sum(axis=1)

TRUE_SIL = peak - 70          # -14.9 dB 以下视为数字静音
print("峰值 %.2f dB  真静音阈值 %.2f dB" % (peak, TRUE_SIL))

sil = e_db < TRUE_SIL
low_bg = (~sil) & (e_db < peak - 28)   # 非静音但低能量
voice = (~sil) & (e_db >= peak - 28)

print("\n=== 时间占比 ===")
print("  真静音         %6.2f%%  (%.1f s)" % (sil.mean() * 100, sil.sum() * hop_s))
print("  低能量背景段   %6.2f%%  (%.1f s)" % (low_bg.mean() * 100, low_bg.sum() * hop_s))
print("  高能量段       %6.2f%%  (%.1f s)" % (voice.mean() * 100, voice.sum() * hop_s))

print("\n=== 低能量背景段（候选 BGM 床）频谱特征 ===")
if low_bg.sum() > 30:
    qS = S[low_bg]
    qe = qS.sum(axis=1)
    print("  该段数 %d  能量中位 %.2f dB（相对峰值 %.1f dB）" % (
        low_bg.sum(), np.median(10 * np.log10(qe + 1e-20)), np.median(e_db[low_bg]) - peak))
    bands = [(0, 60), (60, 150), (150, 400), (400, 1000), (1000, 2500), (2500, 6000), (6000, 12000)]
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        v = qS[:, m].sum() / max(qe.sum(), 1e-20)
        print("    %5d-%5dHz : %6.2f%%  %s" % (lo, hi, v * 100, "#" * int(v * 120)))
    # 谱平坦度
    fl = []
    for i in range(0, len(qS), 10):
        P = qS[i] + 1e-20
        fl.append(np.exp(np.mean(np.log(P))) / np.mean(P))
    print("  谱平坦度中位 %.4f  (<0.02 = 有调性/音乐；>0.08 = 白噪嘶声)" % np.median(fl))
    # 谐波峰数（音乐性）
    prof = qS.mean(axis=0)
    sm = (freqs >= 80) & (freqs <= 4000)
    p = prof[sm]
    pk = np.sum((p[1:-1] > p[:-2]) & (p[1:-1] > p[2:]) & (p[1:-1] > p.mean() * 3))
    print("  80~4000Hz 显著谐波峰 %d 个  (>20 = 音乐/旋律；<8 = 噪声)" % pk)
    # 低频节拍
    fr = S[:, lo_band].sum(axis=1)
    d = fr - fr.mean()
    ac = np.correlate(d, d, mode="full")[len(d) - 1:]
    ac /= (ac[0] + 1e-20)
    lo_lag = max(1, int(1 / 2.0 / hop_s))
    hi_lag = max(lo_lag + 1, int(1 / 4.5 / hop_s))
    seg = ac[lo_lag:hi_lag]
    k = int(np.argmax(seg)) + lo_lag
    print("  低频自相关节拍峰 %.2f Hz  r=%.3f" % (1.0 / (k * hop_s), seg.max()))

print("\n=== 高能量段 vs 低能量段的谱差异（判 BGM 是否铺满）===")
if low_bg.sum() > 30 and voice.sum() > 30:
    for name, mask in [("高能量段", voice), ("低背景段", low_bg)]:
        ms = S[mask]
        me = ms.sum(axis=1)
        v_voice = ms[:, sp_band].sum() / max(me.sum(), 1e-20)
        print("  %s: 300-3400Hz 占比 %.1f%%   低频<200Hz 占比 %.2f%%" % (
            name, v_voice * 100, ms[:, lo_band].sum() / max(me.sum(), 1e-20) * 100))

# 找一段明显的背景床样本，看它像不像音乐
if low_bg.sum() > 30:
    idxs = np.where(low_bg)[0]
    # 找连续段最长的
    print("\n=== 最长连续低能量段（毫秒级）===")
    runs = []
    st = idxs[0]
    prev = idxs[0]
    for i in idxs[1:]:
        if i - prev > 3:
            runs.append((st, prev))
            st = i
        prev = i
    runs.append((st, prev))
    runs.sort(key=lambda r: r[1] - r[0])
    for a, b in runs[:3]:
        print("  %.2f ~ %.2f s (时长 %.2f s)" % (a * hop_s, b * hop_s, (b - a) * hop_s))
    json.dump({"sil_pct": float(sil.mean() * 100), "lowbg_pct": float(low_bg.mean() * 100),
               "high_pct": float(voice.mean() * 100)},
              open("/tmp/refan/bgm_verdict.json", "w"), indent=2)
