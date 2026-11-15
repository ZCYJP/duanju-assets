"""参考片音频分层分析 v2：48k 全频段 + 对白/非对白分段 + BGM 判定"""
import numpy as np
import subprocess
import json
import os

SR = 48000


def decode_stereo(path, sr=SR):
    cmd = ["/usr/local/bin/ffmpeg", "-v", "error", "-i", path, "-vn",
           "-f", "f32le", "-ac", "2", "-ar", str(sr), "-"]
    r = subprocess.run(cmd, capture_output=True)
    d = np.frombuffer(r.stdout, dtype=np.float32).copy()
    return d.reshape(-1, 2)


def db(x):
    v = float(np.max(np.abs(x)))
    return 20 * np.log10(v) if v > 1e-12 else -120.0


def stft(x, sr=SR, nfft=2048, hop=1024):
    if len(x) < nfft:
        return np.zeros((1, nfft // 2 + 1)), np.fft.rfftfreq(nfft, 1.0 / sr)
    n = 1 + (len(x) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(n)[:, None]
    frames = x[idx] * np.hanning(nfft)[None, :]
    return np.abs(np.fft.rfft(frames, axis=1)) ** 2, np.fft.rfftfreq(nfft, 1.0 / sr)


def main():
    st = decode_stereo("/tmp/ref.mp4")
    dur = len(st) / SR
    L, R = st[:, 0], st[:, 1]
    mono = (L + R) / 2.0
    print("=== 参考片音频分层分析 v2 (48kHz) ===")
    print("时长 %.1fs  峰值 %.2f dBFS  RMS %.2f dBFS" % (
        dur, db(mono), db(np.sqrt((mono ** 2).mean()))))

    S, freqs = stft(mono)
    tot = S.sum()
    bands = [(0, 40), (40, 80), (80, 160), (160, 320), (320, 640), (640, 1250),
             (1250, 2500), (2500, 5000), (5000, 8000), (8000, 12000),
             (12000, 16000), (16000, 20000), (20000, 24000)]
    print("\n[全频段能量占比]")
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        v = S[:, m].sum() / max(tot, 1e-12)
        print("  %5d-%5dHz : %6.2f%%  %s" % (lo, hi, v * 100, "#" * int(v * 160)))

    # 帧级能量包络
    fe = S.sum(axis=1)
    fe_db = 10 * np.log10(fe + 1e-20)
    hop_s = 1024 / SR
    print("\n[包络]")
    print("  max %.2f dB  P95 %.2f  P50 %.2f  P10 %.2f  P5 %.2f" % (
        fe_db.max(), np.percentile(fe_db, 95), np.percentile(fe_db, 50),
        np.percentile(fe_db, 10), np.percentile(fe_db, 5)))
    print("  绝对静音(<-60dB相对) 占比 %.2f%%" % ((fe_db < fe_db.max() - 60).mean() * 100))
    print("  P95-P10 动态 %.2f dB   P99-P50 %.2f dB" % (
        np.percentile(fe_db, 95) - np.percentile(fe_db, 10),
        np.percentile(fe_db, 99) - np.percentile(fe_db, 50)))

    # ---- 对白/非对白分段（用 1.5~4kHz 人声频带能量比 vs 全带）----
    speech_band = (freqs >= 300) & (freqs <= 3400)
    sp_ratio = S[:, speech_band].sum(axis=1) / (S.sum(axis=1) + 1e-20)
    voiced = (fe_db > fe_db.max() - 45) & (sp_ratio > np.percentile(sp_ratio, 55))
    # 平滑（>=0.15s 段）
    min_len = int(0.15 / hop_s)
    voiced = np.convolve(voiced.astype(np.float32), np.ones(min_len) / min_len, mode="same") > 0.5

    quiet = (fe_db < fe_db.max() - 40)
    print("\n[对白/非对白]")
    print("  估计对白活跃占比 %.1f%%   安静(非对白)占比 %.1f%%" % (
        voiced.mean() * 100, quiet.mean() * 100))

    # 在非对白段里，是否还有音乐床？检查安静段的频谱形状
    if quiet.sum() > 20:
        qS = S[quiet]
        qb = qS.sum(axis=1)
        print("\n[非对白段（安静区）频谱 —— BGM 床判定]")
        for lo, hi in bands:
            m = (freqs >= lo) & (freqs < hi)
            v = qS[:, m].sum() / max(qb.sum(), 1e-12)
            print("    %5d-%5dHz : %6.2f%%" % (lo, hi, v * 100))
        qflat = []
        for i in range(0, len(qS), 20):
            P = qS[i] + 1e-20
            qflat.append(np.exp(np.mean(np.log(P))) / np.mean(P))
        print("  安静段谱平坦度中位 %.4f (>0.08 = 宽带噪声/嘶声为主)" % np.median(qflat))
        print("  安静段能量中位 %.2f dB（相对峰值）" % (np.median(fe_db[quiet]) - fe_db.max()))
        # 低频节拍周期性
        lowm = freqs < 150
        fr = S[:, lowm].sum(axis=1)
        d = fr - fr.mean()
        ac = np.correlate(d, d, mode="full")[len(d) - 1:]
        ac /= (ac[0] + 1e-20)
        lo_lag = max(1, int((1 / 2.0) / hop_s))
        hi_lag = max(lo_lag + 1, int((1 / 4.5) / hop_s))
        seg = ac[lo_lag:hi_lag]
        k = int(np.argmax(seg)) + lo_lag
        print("  低频(<150Hz)自相关节拍峰 %.2f Hz  r=%.3f  (>0.3 → 有鼓点/BGM律动)" % (
            1.0 / (k * hop_s), seg.max()))
        # 中频音乐性：谐波峰数量
        midS = S[quiet].mean(axis=0)
        sm = (freqs >= 100) & (freqs <= 4000)
        smid = midS[sm]
        peaks = np.sum((smid[1:-1] > smid[:-2]) & (smid[1:-1] > smid[2:]) &
                       (smid[1:-1] > smid[1:-1].mean() * 2))
        print("  安静段 100~4000Hz 显著谐波峰数 %d （>30 → 音乐和声）" % peaks)

    # 立体声宽度
    side = L - R
    c = float(np.corrcoef(L, R)[0, 1])
    print("\n[立体声]")
    print("  L/R 相关系数 %.4f   Side 能量 %.2f%%   L峰值 %.2f  R峰值 %.2f dBFS" % (
        c, (np.mean(side ** 2) / max(np.mean(L ** 2), 1e-12)) * 100, db(L), db(R)))

    # 短时响度波动（LUFS 短时 3s）
    win = int(3.0 / hop_s)
    if len(fe_db) > win:
        sm = np.convolve(fe_db, np.ones(win) / win, mode="valid")
        print("  3s 短时响度波动 std %.2f dB（短剧通常 4~8dB，说明有音乐/音效动态）" % sm.std())

    json.dump({"dur": dur, "peak": db(mono),
               "rms": db(np.sqrt((mono ** 2).mean())),
               "voiced_pct": float(voiced.mean() * 100),
               "quiet_pct": float(quiet.mean() * 100),
               "lr_corr": c,
               "side_pct": float((np.mean(side ** 2) / max(np.mean(L ** 2), 1e-12)) * 100)},
              open("/tmp/refan/audio_v2.json", "w"), indent=2)
    print("\nwrote /tmp/refan/audio_v2.json")


main()
