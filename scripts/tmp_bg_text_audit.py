#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""背景汉字真实性核查（只读，不修改任何原图）

对 /ssd2/duanju/scenes_100/*.png 全部 100 张：
  1) easyocr（GPU，多语言按预期语言匹配 + 全语言兜底）提取文字
  2) tesseract 作为第二引擎交叉验证（含繁体/简/日/韩/英）
  3) 形态学 + 连通域找"文字状区域"，并对每个区域裁剪放大后二次 OCR
     （解决小字/艺术字读不出的问题）
  4) 判定每个字符串是否为真实汉字/日文/韩文/英文（严格 Unicode 规则）
  5) 与 gen_scenes_100.py 预期招牌文本做语义交叉验证
  6) 输出 /ssd2/duanju/scenes_100_text_audit.json + 裁剪图 /ssd2/duanju/bg_text_crops/

判定规则（按需求）:
  允许字符集 = CJK统一表意文字 U+4E00-U+9FFF
             + CJK兼容表意文字 U+F900-U+FAFF
             + 假名 U+3040-U+30FF
             + 谚文 U+AC00-U+D7AF
             + ASCII 字母数字
  出现私用区 U+E000-U+F8FF / 罕见符号 / 其它字符 -> 该字符串判为「伪文字」

图像级最终判定（证据综合）:
  真实文字 = 存在通过字符集检查, 且与预期招牌文本相似度 >= 0.5 的字符串
  伪文字   = 检出文字状区域, 但没有可匹配预期文本的真实字符串
  无文字   = 完全未检出文字状区域且无任何 OCR 结果
"""
import difflib
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict

os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import cv2
import numpy as np
import pytesseract
from PIL import Image

ROOT = "/ssd2/duanju/scenes_100"
CROP_DIR = "/ssd2/duanju/bg_text_crops"
OUT_JSON = "/ssd2/duanju/scenes_100_text_audit.json"
SRC_SCENES = "/ssd2/duanju/scripts/gen_scenes_100.py"

EXPECT_SIZE = (832, 1248)
MATCH_THRESHOLD = 0.5

ALLOWED_RANGES = [
    (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x3040, 0x30FF), (0xAC00, 0xD7AF),
]

# easyocr reader 组合: ch_sim 与 ch_tra 不能共存
READER_SPECS = {
    "sim": ["ch_sim", "en"],
    "tra": ["ch_tra", "en"],
    "ja": ["ja", "en"],
    "ko": ["ko", "en"],
}
# 预期语言 -> 首选 reader
LANG_TO_READER = {
    "简体中文": "sim", "繁体中文": "tra", "日语": "ja",
    "韩语": "ko", "英语": "sim", "法语": "sim",
}
TESS_LANG = {
    "sim": "chi_sim+eng", "tra": "chi_tra+eng", "ja": "jpn+eng",
    "ko": "kor+eng", "en": "eng",
}


def _in_ranges(cp):
    return any(lo <= cp <= hi for lo, hi in ALLOWED_RANGES)


def classify_chars(s):
    """返回 (verdict, bad_chars, bad_codepoints) —— 严格 Unicode 规则

    空白字符（空格 U+0020 / 制表 U+0009 / 全角空格 U+3000）视为合法：
    多词招牌（"DRIVE SAFE"、"Salon de Beauté"）的词间空格不是伪文字。
    其余非字母数字符号（'-' '|' '—' 等）仍判伪。
    """
    stripped = s.strip()
    if not stripped:
        return "empty", [], []
    bad = []
    for ch in stripped:
        cp = ord(ch)
        if ch.isspace():
            continue
        if ch.isascii() and ch.isalnum():
            continue
        if _in_ranges(cp):
            continue
        bad.append(ch)
    return ("pseudo" if bad else "real"), bad, [f"U+{ord(c):04X}" for c in bad]


def char_detail(s):
    """逐字符明细，便于人工复核"""
    out = []
    for ch in s:
        cp = ord(ch)
        if ch.isspace():
            k = "space"
        elif ch.isascii() and ch.isalnum():
            k = "ascii_alnum"
        elif _in_ranges(cp):
            k = "cjk_or_kana_or_hangul"
        elif 0xE000 <= cp <= 0xF8FF:
            k = "private_use"
        else:
            k = f"other_{unicodedata.category(ch)}"
        out.append({"ch": ch, "cp": f"U+{cp:04X}", "kind": k})
    return out


def load_expected():
    src = open(SRC_SCENES, encoding="utf-8").read()
    m = re.search(r"SCENES = \[(.*?)\n\]", src, re.S)
    if not m:
        return {}
    ns = {}
    exec("SCENES = [" + m.group(1) + "\n]", ns)
    return {r[0]: {"cn": r[1], "en": r[2], "expected": r[3] or "", "lang": r[4]}
            for r in ns["SCENES"]}


# ------------------------------------------------------------ 形态学文字区域
def detect_text_regions(img):
    g = np.array(img.convert("L"))
    h, w = g.shape
    scale = h / 1248.0
    regions = []
    edges = cv2.Canny(g, 60, 180)
    _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for mask in (edges, otsu, cv2.bitwise_not(otsu)):
        n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
        comps = []
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if ch < 6 * scale or ch > 200 * scale:
                continue
            if cw > 0.6 * w or area > 0.04 * h * w:
                continue
            ar = cw / float(ch)
            if ar > 6.0 or ar < 0.08:
                continue
            fill = area / float(cw * ch)
            if fill < 0.06 or fill > 0.98:
                continue
            comps.append((x, y, cw, ch, area))
        if not comps:
            continue
        comps.sort(key=lambda c: c[1])
        rows = []
        for c in comps:
            placed = False
            for row in rows:
                ry0 = min(r[1] for r in row)
                ry1 = max(r[1] + r[3] for r in row)
                ryc = (ry0 + ry1) / 2.0
                chc = c[1] + c[3] / 2.0
                if abs(chc - ryc) < max(c[3], (ry1 - ry0) * 0.5) * 0.9:
                    row.append(c)
                    placed = True
                    break
            if not placed:
                rows.append([c])
        for row in rows:
            if len(row) < 3:
                continue
            row.sort(key=lambda c: c[0])
            hs = np.array([c[3] for c in row], dtype=float)
            if hs.mean() < 8 * scale:
                continue
            if hs.std() / max(hs.mean(), 1e-6) > 0.55:
                continue
            gaps = [row[i + 1][0] - (row[i][0] + row[i][2]) for i in range(len(row) - 1)]
            if gaps and max(gaps) > 3.2 * hs.mean():
                continue
            x0 = min(c[0] for c in row); y0 = min(c[1] for c in row)
            x1 = max(c[0] + c[2] for c in row); y1 = max(c[1] + c[3] for c in row)
            rw, rh = x1 - x0, y1 - y0
            if rw < 25 * scale or rh < 8 * scale:
                continue
            if rw > 0.97 * w and rh > 0.9 * h:
                continue
            regions.append({"box": (int(x0), int(y0), int(rw), int(rh)),
                            "n_comp": len(row), "h_mean": float(hs.mean()),
                            "h_std_ratio": float(hs.std() / max(hs.mean(), 1e-6))})
    regions.sort(key=lambda r: -r["n_comp"])
    kept = []
    for r in regions:
        x, y, ww, hh = r["box"]
        dup = False
        for k in kept:
            kx, ky, kw, kh = k["box"]
            ix = max(0, min(x + ww, kx + kw) - max(x, kx))
            iy = max(0, min(y + hh, ky + kh) - max(y, ky))
            inter = ix * iy
            union = ww * hh + kw * kh - inter
            if union > 0 and inter / union > 0.35:
                dup = True
                break
        if not dup:
            kept.append(r)
    return kept[:10]


# ------------------------------------------------------------ OCR 引擎
READERS = {}


def init_readers(use_gpu=True):
    global READERS
    import easyocr
    for k, langs in READER_SPECS.items():
        READERS[k] = easyocr.Reader(langs, gpu=use_gpu, verbose=False)
    print(f"[engine] easyocr readers ready: {list(READERS)}", flush=True)


def easyocr_full(img_arr, key):
    """整图 easyocr，返回 (strings, boxes)"""
    r = READERS[key]
    try:
        res = r.readtext(img_arr, detail=1, mag_ratio=1.5,
                         text_threshold=0.6, low_text=0.35,
                         bbox_min_score=0.15, canvas_size=3000)
    except Exception as e:
        print(f"    ! easyocr full error {key}: {e}", flush=True)
        return [], []
    strings, boxes = [], []
    for quad, txt, conf in res:
        txt = (txt or "").strip()
        if not txt:
            continue
        pts = np.array(quad, dtype=float)
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        boxes.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
        strings.append({"text": txt, "conf": float(conf),
                        "box": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                        "engine": f"easyocr:{key}"})
    return strings, boxes


def easyocr_crop(img_arr, box, key):
    x, y, w, h = box
    H, W = img_arr.shape[:2]
    pad = 4
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return []
    sub = img_arr[y0:y1, x0:x1]
    try:
        up = cv2.resize(sub, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    except Exception:
        up = sub
    try:
        res = READERS[key].readtext(up, detail=1, paragraph=False,
                                    text_threshold=0.5, low_text=0.3)
    except Exception:
        return []
    out = []
    for quad, txt, conf in res:
        txt = (txt or "").strip()
        if not txt:
            continue
        out.append({"text": txt, "conf": float(conf), "box": None,
                    "engine": f"easyocr_crop:{key}"})
    return out


def tesseract_on(arr, lang, tag):
    out = []
    try:
        data = pytesseract.image_to_data(arr, lang=lang, output_type=pytesseract.Output.DICT,
                                         config="--oem 3 --psm 11")
    except Exception as e:
        print(f"    ! tess error {tag}: {e}", flush=True)
        return out
    for i in range(len(data.get("text", []))):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if conf < 30:
            continue
        out.append({"text": txt, "conf": conf, "box": None,
                    "engine": f"tesseract:{tag}"})
    return out


# ------------------------------------------------------------ 工具
def norm(s):
    return re.sub(r"\s+", "", s or "")


def similarity(a, b):
    """严格相似度：min(序列相似比, 预期串覆盖率)。

    覆盖�� = 预期串中被识别串匹配到的字符数 / 预期串长度
    用 min 而非 ratio 是为了避免 'uty' 这类子串把 'Beauty Studio' 判成 1.0
    """
    if not a or not b:
        return 0.0
    m = difflib.SequenceMatcher(None, a, b)
    ratio = m.ratio()
    cov = sum(blk.size for blk in m.get_matching_blocks()) / float(len(a))
    return min(ratio, cov)


def save_crop(img, box, path, pad=6):
    x, y, w, h = box
    W, H = img.size
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return False
    img.crop((x0, y0, x1, y1)).save(path)
    return True


def dedup_regions(boxes, limit=10):
    """按 IoU 去重文字区域"""
    out = []
    for b in sorted(boxes, key=lambda t: -(t[2] * t[3])):
        if len(out) >= limit:
            break
        dup = False
        for k in out:
            ix = max(0, min(b[0] + b[2], k[0] + k[2]) - max(b[0], k[0]))
            iy = max(0, min(b[1] + b[3], k[1] + k[3]) - max(b[1], k[1]))
            inter = ix * iy
            union = b[2] * b[3] + k[2] * k[3] - inter
            if union > 0 and inter / union > 0.4:
                dup = True
                break
        if not dup:
            out.append(b)
    return out


EXPECTED = {}


def audit_one(path):
    sid = os.path.splitext(os.path.basename(path))[0]
    exp_info = EXPECTED.get(sid, {})
    exp_text = exp_info.get("expected", "")
    exp_lang = exp_info.get("lang", "")
    primary = LANG_TO_READER.get(exp_lang, "sim")

    img = Image.open(path)
    img.load()
    W, H = img.size
    gray = np.array(img.convert("L"))
    rgb = np.array(img.convert("RGB"))

    rec = {
        "file": os.path.basename(path), "path": path, "size": [W, H],
        "size_ok": (W, H) == EXPECT_SIZE,
        "expected": exp_text, "lang": exp_lang,
        "ocr_strings": [], "pseudo_strings": [], "real_strings": [],
        "text_regions": [], "n_text_regions": 0, "crops": [],
        "verdict": "无文字", "reason": "", "similarity": 0.0,
        "best_match": "", "matched": False, "engines_used": [],
    }

    # 1) 形态学文字区域
    morph = detect_text_regions(img)
    rec["n_text_regions"] = len(morph)
    for r in morph:
        rec["text_regions"].append({"box": list(r["box"]), "n_comp": r["n_comp"],
                                    "h_mean": round(r["h_mean"], 1),
                                    "h_std_ratio": round(r["h_std_ratio"], 3),
                                    "src": "morph"})

    # 2) easyocr 整图: 首选 reader + 其余 reader 兜底
    all_strings = []
    all_boxes = []
    order = [primary] + [k for k in READER_SPECS if k != primary]
    for key in order:
        s, b = easyocr_full(rgb, key)
        all_strings += s
        all_boxes += b
        rec["engines_used"].append(f"easyocr:{key}")
    # 3) tesseract 整图: 首选 + 英文 + 日文
    tess_langs = [TESS_LANG.get(primary, "eng")]
    for extra in ("chi_sim+eng", "chi_tra+eng", "jpn+eng", "kor+eng"):
        if extra not in tess_langs:
            tess_langs.append(extra)
    for tl in tess_langs:
        all_strings += tesseract_on(gray, tl, tl)
        rec["engines_used"].append(f"tesseract:{tl}")

    # 4) 区域二次 OCR（easyocr 检测框 + 形态学框）
    region_boxes = dedup_regions(all_boxes + [r["box"] for r in morph], limit=12)
    for box in region_boxes:
        for key in (primary, "sim", "tra"):
            all_strings += easyocr_crop(rgb, box, key)
    for r in morph[:6]:
        x, y, w, h = r["box"]
        sub = gray[max(0, y):y + h, max(0, x):x + w]
        if sub.size == 0:
            continue
        up = cv2.resize(sub, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        all_strings += tesseract_on(up, TESS_LANG.get(primary, "eng"), "crop")

    # 5) 去重 + 分类
    seen = set()
    entries = []
    for s in all_strings:
        t = s["text"]
        if not t.strip():
            continue
        key = (t, s["engine"])
        if key in seen:
            continue
        seen.add(key)
        v, bad, cps = classify_chars(t)
        e = dict(s)
        e["char_verdict"] = v
        if bad:
            e["bad_chars"] = bad[:12]
            e["bad_codepoints"] = cps[:12]
        entries.append(e)
    rec["ocr_strings"] = entries
    rec["pseudo_strings"] = [e["text"] for e in entries if e["char_verdict"] == "pseudo"]
    rec["real_strings"] = [e["text"] for e in entries if e["char_verdict"] == "real"]

    # 6) 语义匹配（严格：覆盖率与相似比取 min，避免子串误判）
    exp_n = norm(exp_text)
    best_sim, best_txt = 0.0, ""
    if exp_n:
        for e in entries:
            c = norm(e["text"])
            if not c:
                continue
            s = similarity(exp_n, c)
            if s > best_sim:
                best_sim, best_txt = s, c
    rec["similarity"] = round(best_sim, 4)
    rec["best_match"] = best_txt
    rec["matched"] = best_sim >= MATCH_THRESHOLD

    # 7) 裁剪图保存
    cropdir = os.path.join(CROP_DIR, sid)
    os.makedirs(cropdir, exist_ok=True)
    for i, box in enumerate(region_boxes[:8]):
        p = os.path.join(cropdir, f"region_{i:02d}.png")
        if save_crop(img, box, p):
            rec["crops"].append({"file": p, "kind": "region", "box": list(box)})
    for i, r in enumerate(morph[:5]):
        p = os.path.join(cropdir, f"morph_{i:02d}.png")
        if save_crop(img, r["box"], p):
            rec["crops"].append({"file": p, "kind": "morph", "n_comp": r["n_comp"]})

    # 8) 最终判定
    has_text_signal = bool(entries) or rec["n_text_regions"] > 0
    # 只有通过字符集检查、且严格覆盖预期文本 >= MATCH_THRESHOLD 的串才算"真实文字"证据
    real_match = None
    if exp_n:
        for e in entries:
            if e["char_verdict"] != "real":
                continue
            c = norm(e["text"])
            if not c:
                continue
            s = similarity(exp_n, c)
            if s >= MATCH_THRESHOLD:
                if real_match is None or s > real_match[1]:
                    real_match = (e["text"], s, e["engine"])

    if real_match:
        rec["verdict"] = "真实文字"
        rec["reason"] = (f"识别出与预期招牌匹配的字符串 {real_match[0]!r} "
                         f"(覆盖相似度={real_match[1]:.2f}, {real_match[2]})")
        rec["similarity"] = round(real_match[1], 4)
        rec["best_match"] = real_match[0]
        rec["matched"] = True
    elif not has_text_signal:
        rec["verdict"] = "无文字"
        rec["reason"] = "OCR 无结果且形态学未检出文字状区域"
    else:
        n_pseudo = len(rec["pseudo_strings"])
        if rec["n_text_regions"] > 0:
            rec["verdict"] = "伪文字"
            rec["reason"] = (f"形态学检出 {rec['n_text_regions']} 处文字状区域, "
                             f"但 OCR 结果均无法匹配预期招牌 {exp_text!r} "
                             f"(最高覆盖相似度 {best_sim:.2f})")
        else:
            rec["verdict"] = "伪文字"
            rec["reason"] = (f"OCR 结果均无法匹配预期招牌 {exp_text!r} "
                             f"(最高覆盖相似度 {best_sim:.2f}, 伪串 {n_pseudo} 个)")
    return rec


def main():
    global EXPECTED
    EXPECTED = load_expected()
    files = sorted(os.path.join(ROOT, f) for f in os.listdir(ROOT)
                   if f.lower().endswith(".png"))
    os.makedirs(CROP_DIR, exist_ok=True)
    print(f"[audit] {len(files)} 张背景图", flush=True)
    init_readers(use_gpu=True)

    results = []
    for i, p in enumerate(files):
        try:
            rec = audit_one(p)
        except Exception as e:
            import traceback
            traceback.print_exc()
            rec = {"file": os.path.basename(p), "path": p, "verdict": "无文字",
                   "reason": f"处理异常: {e}", "ocr_strings": [], "crops": [],
                   "n_text_regions": 0, "similarity": 0.0, "expected": "",
                   "pseudo_strings": [], "real_strings": [], "text_regions": [],
                   "matched": False, "best_match": ""}
        results.append(rec)
        mark = {"真实文字": "OK", "伪文字": "!!", "无文字": "--"}[rec["verdict"]]
        print(f"  {mark} [{i+1}/{len(files)}] {rec['file']:<26} "
              f"sim={rec.get('similarity',0):.2f} reg={rec.get('n_text_regions',0):>2} "
              f"got={rec.get('best_match','')!r} exp={rec.get('expected','')!r}", flush=True)

    summary = {"total": len(results)}
    for k in ("真实文字", "伪文字", "无文字"):
        summary[k] = sum(1 for r in results if r.get("verdict") == k)
    summary["伪文字图片"] = [r["file"] for r in results if r.get("verdict") == "伪文字"]
    summary["无文字图片"] = [r["file"] for r in results if r.get("verdict") == "无文字"]
    summary["尺寸异常"] = [r["file"] for r in results if not r.get("size_ok", True)]

    out = {
        "meta": {
            "ocr_engines": ["easyocr 1.7.2 (GPU, ch_sim/ch_tra/ja/ko/en 独立 reader)",
                            "tesseract 4.1.1 (chi_sim/chi_tra/jpn/kor/eng)"],
            "scenes_dir": ROOT, "crop_dir": CROP_DIR,
            "expected_source": SRC_SCENES,
            "match_threshold": MATCH_THRESHOLD,
            "rule": ("允许字符: CJK统一 U+4E00-U+9FFF / CJK兼容 U+F900-U+FAFF / "
                     "假名 U+3040-U+30FF / 谚文 U+AC00-U+D7AF / ASCII字母数字; "
                     "私用区与罕见符号判为伪文字"),
        },
        "summary": summary,
        "items": results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n[summary] " + json.dumps({k: v for k, v in summary.items()
                                       if not isinstance(v, list)}, ensure_ascii=False), flush=True)
    print(f"[summary] 伪文字({len(summary['伪文字图片'])}): {summary['伪文字图片']}", flush=True)
    print(f"[summary] 无文字({len(summary['无文字图片'])}): {summary['无文字图片']}", flush=True)
    print(f"[done] -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
