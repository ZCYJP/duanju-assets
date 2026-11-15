#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""情绪音色音高回正 v2：迭代式把漂移音高压回角色基线附近。

问题根因（v3 实测）
--------------------
Qwen3-TTS 的 VoiceDesign 在「喜 / 怒 / 惊」这类高唤醒情绪下会自发拔高音高：
男声角色实测 f0 达 260~308Hz，而男声基线只有 ~117Hz —— 已是女声音域，
构成性别漂移。改 prompt 只能缓解（实测 r 2.43 -> 2.05），
候选池里拿不到完全达标的低音高候选，必须后期回正。

解法：相位声码器变调（pitch shift）
----------------------------------
  resample 降调（音高 ×factor，时长 ×1/factor）
    -> phase_vocoder 把时长压回原长（音高不变）
  音高精确到目标，时长与语速完全不变。

v2 相对 v1 的改进
-----------------
1. 【迭代】v1 单次 factor 限幅 [0.70,1.30]，但漂移可能到 r=2.5，
   需要 factor≈0.5 —— 一次大幅降调音质劣化明显。
   v2 改为多轮迭代，每轮限幅 [0.80,1.25]，最多 6 轮，
   逐步逼近目标，累积可覆盖 r 3.0+ 的极端漂移。
2. 【时长精确】v1 的 istft 输出长度与原长有毫秒级偏差，
   v2 显式 pad/truncate 到原始样本数。
3. 【不原地覆盖】v1 直接写回原文件，出错无法回滚。
   v2 默认输出到独立目录 --out-dir，验收通过后再合并。
4. 【可回退】若某条回正后音高仍未落入目标区间，标记 not_in_range 供复核。

踩过的坑（保留注释，避免重犯）
------------------------------
1. librosa 0.11.0 的 effects.time_stretch 是坏的（stft 移除了 sr 参数，
   但 time_stretch 内部仍传 sr=）→ 必须自己串 stft→phase_vocoder→istft。
2. phase_vocoder 的 rate 语义与直觉相反：rate<1 变慢，rate>1 变快。
   resample 后长 1/factor 倍，压回原长需 rate=1/factor（>1=变快）。
3. librosa.resample 的 target_sr 降低会【升高】音高，
   降调必须用 target_sr = sr / factor。
4. istft 不能传 length=len(y)（那会把拉伸结果又裁回原长，拉伸白做）。

用法
----
  python3 tmp_emotion_pitch_fix.py --emotions xi,nu,jing --dry-run
  python3 tmp_emotion_pitch_fix.py --emotions xi,nu,jing \
      --out-dir /ssd2/duanju/voices/emotions_fixed
  python3 tmp_emotion_pitch_fix.py --merge          # 合并回原目录
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
import soundfile as sf
import librosa

EMOTIONS = "/ssd2/duanju/voices/emotions"
BASE_F0 = "/ssd2/duanju/voices/base_f0.json"
CAST = "/ssd2/duanju/scripts/cast_all_150.json"

# 各情绪的目标音高倍率区间（相对角色基线）——与 gen_emotion_voices_v2.py 一致
TARGET_RATIO = {
    "xi":   (1.00, 1.30),
    "nu":   (0.95, 1.30),
    "ai":   (0.82, 1.02),
    "le":   (0.95, 1.12),
    "ju":   (1.00, 1.30),
    "jing": (1.05, 1.40),
    "yan":  (0.85, 1.05),
    "you":  (0.90, 1.08),
    "xiu":  (0.98, 1.22),
}

# 单轮变调限幅（比 v1 更保守，配合多轮迭代）
# 注意：0.80^8 = 0.17，8 轮最多累计降到 17%，
# 足以覆盖 r=3.13（需要 0.32）这类极端漂移；
# 但实测 M02 r=3.13 用 8 轮只到 2.31 —— 因为每轮的实际降调幅度
# 小于名义 factor（f0 估计误差会累积），
# 所以这里把限幅放宽到 0.75，让每轮更有力。
STEP_LO, STEP_HI = 0.75, 1.30
MAX_ROUNDS = 10
# 判定「无需处理」的容差
EPS = 0.03
# 闭环收敛容差（倍率）
TOL = 0.03


def f0_median(x, sr, octave_fix=False):
    """中位基频：自相关。

    【口径必须与 base_f0.json 一致】
    基线 base_f0.json 由 tmp_extract_base_f0.py 用【自相关 + 70~400Hz 搜索窗、
    无倍频修正】提取。四算法交叉验证（tmp_f0_cross.py，150 条真实情绪音频）：
        ac    mean=1.67  median=1.48
        yin   mean=1.13  median=1.09   （比 ac 低 30%，66.7% 样本差 >20%）
        pyin  mean=1.98  median=1.64   （比 ac 高 17%）
    若回正用 yin 而基线用 ac，等于拿两把不同的尺子量同一件东西，
    会把音频"修"到错误位置。所以默认口径与基线严格一致。

    【octave_fix 的作用】
    自相关在大幅降调后（基频低于 70Hz 搜索下限时）会锁到 2/3 倍谐波，
    测出的 f0 反而变高 -> 闭环发散（实测 M65 从 2.84 变到 4.69）。
    octave_fix=True 时做倍频折叠修正，用于【校验】而非主口径：
    修正逻辑对正常音频影响 <0.2%（150 条中 |diff|>20% 比例为 0%）。
    """
    frame = int(sr * 0.04)
    hop = int(sr * 0.02)
    # 搜索范围必须与基线口径严格一致：70~400Hz。
    #
    # 【为什么不能放宽上限】
    # 试过放宽到 700Hz，结果基线读数全乱：
    #   M03 裴九渊  400Hz上限=121.5Hz，700Hz上限=421.1Hz（差 246%）
    #   M05 顾怀砚  400Hz上限= 90.4Hz，700Hz上限=280.8Hz（差 211%）
    # 原因是放宽上限后，自相关会锁到 2/3 倍谐波上（主峰与谐波峰竞争）。
    # 基线 base_f0.json 用 400Hz 上限算的，若回正用 700Hz，
    # 等于拿两把尺子量同一件东西，倍率全错。
    #
    # 【代价与应对】
    # 代价：漂移音频的 f0 超过 400Hz 时读数被截断，初始倍率偏小。
    # 应对：用【开环一次性】降调（不依赖初始读数的精确性），
    #       降到测量可信区后再用闭环精修。
    lo, hi = int(sr / 400), int(sr / 70)
    vals = []
    for st in range(0, len(x) - frame, hop):
        seg = x[st:st + frame].astype(np.float64)
        if np.sqrt(np.mean(seg ** 2)) < 0.012:
            continue
        seg = seg - seg.mean()
        ac = np.correlate(seg, seg, "full")[frame - 1:]
        if hi >= len(ac):
            continue
        seg_ac = ac[lo:hi]
        if len(seg_ac) == 0:
            continue
        k = int(np.argmax(seg_ac))
        lag = lo + k
        f = sr / lag
        if octave_fix:
            peak = seg_ac[k]
            half_pos = len(seg_ac) // 2
            half_peak = seg_ac[half_pos] if half_pos < len(seg_ac) else 0.0
            if half_peak > 0.85 * peak and f > 82.5:
                f = f / 2.0
        vals.append(f)
    return float(np.median(vals)) if vals else 0.0


def pitch_shift_once(x, sr, factor):
    """单轮变调：factor<1 降调、>1 升调；返回与输入等长的波形。

    单位测试（tmp_pitch_unit.py）已验证：对 220Hz 正弦，
      音高倍率 = factor（0.801 / 0.602，误差 <0.3%）
      时长倍率 = 1.000

    【幅度补偿】重要！
    phase_vocoder + istft 重建的波形幅度会明显低于输入
    （实测 rms 掉到 0.42 倍 ≈ -7.6dB）。
    原因：istft 的窗口重叠相加在相位失配时发生相消干涉，
    能量被抵消。若不补偿，回正后的音频会明显偏轻，
    在混音里表现为"这条情绪比其它情绪小声一截"。
    所以每次变调后都必须把 rms 拉回原值。
    """
    if abs(factor - 1.0) < 0.01:
        return x.astype(np.float32)
    n_in = len(x)
    rms_in = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    y = librosa.resample(x.astype(np.float32), orig_sr=sr,
                         target_sr=max(int(sr / factor), 1000))
    D = librosa.stft(y, n_fft=1024, hop_length=256)
    # resample 后长 1/factor 倍，压回原长需 rate = 1/factor（>1 = 变快）
    Ds = librosa.phase_vocoder(D, rate=1.0 / factor, hop_length=256, n_fft=1024)
    out = np.asarray(librosa.istft(Ds, hop_length=256, n_fft=1024), dtype=np.float32)
    # 时长精确保留：与原长对齐
    if len(out) < n_in:
        out = np.pad(out, (0, n_in - len(out)))
    elif len(out) > n_in:
        out = out[:n_in]
    # 幅度补偿：把 rms 拉回输入水平
    rms_out = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    if rms_out > 1e-8 and rms_in > 1e-8:
        out = out * (rms_in / rms_out)
    # 峰值约束：补偿后不能削波
    pk = float(np.max(np.abs(out)))
    if pk > 0.99:
        out = out * (0.99 / pk)
    return out


def pitch_shift_to(x, sr, base_f0, target_r, lo_r, hi_r, max_rounds=8):
    """变调到目标倍率。

    策略演进（都是实测逼出来的）
    ------------------------------
    v1 单次开环：按理论倍率算一次。
      失败：真实语音的 f0 估计有 7~10% 误差，一次算不准。

    v2 迭代闭环：变调 -> 实测 -> 修正，直到收敛。
      中等漂移（r 1.5~2.5）有效，但极端漂移（r>3.0）发散：
        M61 r 3.45 -> 3.90（反而升高）、M02 3.13 -> 3.76
      根因（合成信号基准实验 tmp_pitch_probe2.py 定位）：
        220Hz 正弦升到 3.45 倍后应为 759Hz，但自相关只测到 381Hz
        —— 搜索上限 400Hz 撞顶，**测量本身已经失真**。
        闭环据此认为"还降得不够"，继续降调，越降越错。

    v3 一次性 + 精修（本版）：
      第一步用【开环一次性】把倍率拉到目标附近 ——
      即使 f0 测量在极端区不准，开环也不会被错误测量带偏；
      实测极端案例从"发散到 4.5~5.0"改善为"收敛到 1.43~1.89"，
      且用时更少、累积失真更小。
      第二步若仍有残差且倍率已回到测量可信区（<2.5），
      再做闭环精修，此时测量可信，能精确到位。
    """
    y = x.astype(np.float32)
    rounds = 0

    f0_now = f0_median(y, sr, octave_fix=True)
    r_now = f0_now / base_f0 if base_f0 else 0.0

    # ---- 第一步：粗降（开环）----
    # 初始读数在极端区被 400Hz 上限截断（偏小），
    # 若按它算 need，降调量会不足。所以这里用一个【保守的固定步长】
    # 反复粗降，直到读数落进可信区（r < 2.0）——
    # 此时读数可信，再交给精修阶段。
    # 宁可先降过头（后面能升回来），也不能降不够（读数不可信无法判断）。
    if r_now > 2.0:
        # 用 0.72 的固定步长粗降，每轮重新测量；
        # 当读数进入可信区就停。
        for _ in range(6):
            if r_now <= 2.0:
                break
            y = pitch_shift_once(y, sr, 0.72)
            rounds += 1
            f0_now = f0_median(y, sr, octave_fix=True)
            r_now = f0_now / base_f0 if base_f0 else 0.0

    # ---- 第二步：精修（闭环，仅在测量可信区）----
    # r_now < 2.0 时自相关读数可信（未撞 400Hz 上限），
    # 此时闭环可以精确收敛到目标。
    prev_r = r_now
    while rounds < max_rounds and 0 < r_now < 2.5:
        if abs(r_now - target_r) <= TOL:
            break
        # 防发散：测得倍率反而升高 -> 锁到谐波，停止
        if r_now > prev_r * 1.02:
            break
        prev_r = r_now
        need = target_r / r_now
        step = min(max(need, STEP_LO), STEP_HI)
        if abs(step - 1.0) < 0.01:
            break
        y = pitch_shift_once(y, sr, step)
        rounds += 1
        f0_now = f0_median(y, sr, octave_fix=True)
        r_now = f0_now / base_f0 if base_f0 else 0.0

    return y, rounds, r_now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emotions", default="xi,nu,jing")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-dir", default="",
                    help="回正后输出目录；为空则原地覆盖（不推荐）")
    ap.add_argument("--merge", action="store_true",
                    help="把 out-dir 下的结果合并回原目录")
    ap.add_argument("--out", default="/ssd2/duanju/voices/pitch_fix_report_v2.json")
    args = ap.parse_args()

    if args.merge:
        src = args.out_dir or "/ssd2/duanju/voices/emotions_fixed"
        n = 0
        for p in glob.glob(f"{src}/*/*.wav"):
            dst = os.path.join(EMOTIONS, os.path.relpath(p, src))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(p, dst)
            n += 1
        print(f"[merge] {n} 个文件已合并回 {EMOTIONS}")
        return

    base_map = json.load(open(BASE_F0, encoding="utf-8"))
    gender_map = {c["id"]: c.get("gender", "") for c in
                  json.load(open(CAST, encoding="utf-8"))}
    emos = [e.strip() for e in args.emotions.split(",") if e.strip()]
    out_dir = args.out_dir or EMOTIONS

    all_report = []
    for ek in emos:
        lo_r, hi_r = TARGET_RATIO.get(ek, (0.9, 1.3))
        files = sorted(glob.glob(f"{EMOTIONS}/*/*_{ek}.wav"))
        fixed = 0
        in_range_before = 0
        in_range_after = 0
        rounds_hist = []
        report = []
        for p in files:
            cid = os.path.basename(p).split("_")[0]
            base = base_map.get(cid, 0.0)
            if base <= 0:
                continue
            try:
                x, sr = sf.read(p, dtype="float32")
                if x.ndim > 1:
                    x = x[:, 0]
                f0_before = f0_median(x, sr, octave_fix=True)
                r_before = f0_before / base
                dur_before = len(x) / sr
                in_range_before += int(lo_r - TOL <= r_before <= hi_r + TOL)

                if lo_r - TOL <= r_before <= hi_r + TOL:
                    # 已达标：不动（但仍写入 out_dir 以便整目录替换）
                    if out_dir != EMOTIONS:
                        dst = os.path.join(out_dir, os.path.relpath(p, EMOTIONS))
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        if not args.dry_run:
                            shutil.copy2(p, dst)
                    in_range_after += 1
                    report.append({"path": p, "cid": cid, "emotion": ek,
                                   "changed": False, "base": base,
                                   "f0_before": round(f0_before, 1),
                                   "ratio_before": round(r_before, 3),
                                   "dur": round(dur_before, 3)})
                    continue

                # 目标：区间中值（保留 9 情绪之间的音高层次，
                # 只压到下界会让所有情绪音高挤成一档，失去区分度）
                target_r = (lo_r + hi_r) / 2.0
                y, rounds, r_after = pitch_shift_to(
                    x, sr, base, target_r, lo_r, hi_r)
                rounds_hist.append(rounds)

                pk = float(np.max(np.abs(y)))
                if pk > 0.97:
                    y = y * (0.97 / pk)

                # 最终响度对齐：确保回正后响度与原文件一致
                # （多轮变调后可能有累积偏差）
                #
                # 【为什么用 rms 而不是 p95】
                # 相位声码器变调会降低波形的相干性，峰值相对 rms 下降
                # （crest factor 变小，实测 2.40 -> 2.12）。
                # 若按 p95 对齐，放大系数会把峰值推出 0.97 限幅，
                # 触发硬压缩后 p95 反而掉到 0.81x —— 比不对齐更糟。
                # 按 rms 对齐 + 软限幅，响度误差可控制在 1dB 内。
                rms_in = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
                rms_out = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
                if rms_out > 1e-8 and rms_in > 1e-8:
                    y = y * (rms_in / rms_out)
                    pk = float(np.max(np.abs(y)))
                    if pk > 0.99:
                        # 软限幅：tanh 饱和而非线性硬削，保留瞬态感
                        # 线性缩放会把整条音频一起压低（连带 rms 掉下去）
                        over = pk / 0.99
                        y = np.tanh(y * (over / 0.99) / over) * 0.99
                        # tanh 压缩后重新校准 rms
                        rms_out2 = float(
                            np.sqrt(np.mean(y.astype(np.float64) ** 2)))
                        if rms_out2 > 1e-8:
                            y = y * (rms_in / rms_out2)
                            pk = float(np.max(np.abs(y)))
                            if pk > 0.99:
                                y = y * (0.99 / pk)

                f0_after = f0_median(y, sr, octave_fix=True)
                r_after = f0_after / base
                in_range_after += int(lo_r - TOL <= r_after <= hi_r + TOL)

                dst = os.path.join(out_dir, os.path.relpath(p, EMOTIONS))
                if out_dir != EMOTIONS:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                if not args.dry_run:
                    sf.write(dst, y, sr)

                fixed += 1
                report.append({
                    "path": p, "dst": dst, "cid": cid, "emotion": ek,
                    "gender": gender_map.get(cid, ""),
                    "changed": True, "rounds": rounds,
                    "factor_total": round(target_r / r_before, 3),
                    "f0_before": round(f0_before, 1),
                    "f0_after": round(f0_after, 1), "base": base,
                    "ratio_before": round(r_before, 3),
                    "ratio_after": round(r_after, 3),
                    "in_range": bool(lo_r - TOL <= r_after <= hi_r + TOL),
                    "dur_before": round(dur_before, 3),
                    "dur_after": round(len(y) / sr, 3),
                })
            except Exception as ex:
                report.append({"path": p, "cid": cid, "emotion": ek,
                               "error": f"{type(ex).__name__}: {str(ex)[:120]}"})

        rr = [r for r in report if "ratio_before" in r]
        if rr:
            rb = np.array([r["ratio_before"] for r in rr])
            ra = np.array([r.get("ratio_after", r["ratio_before"]) for r in rr])
            print(f"[done] {ek}: 处理 {fixed}/{len(files)}  "
                  f"倍率均值 {rb.mean():.2f}->{ra.mean():.2f}  "
                  f"达标 {in_range_before}->{in_range_after}"
                  f"  轮次均值 {np.mean(rounds_hist) if rounds_hist else 0:.1f}",
                  flush=True)
        all_report += report

    json.dump(all_report, open(args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
