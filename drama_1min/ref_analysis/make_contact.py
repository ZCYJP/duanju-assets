"""拼参考片典型帧 contact sheet + 我们成片的对照帧"""
import os
from PIL import Image, ImageDraw, ImageFont

F_HI = "/tmp/refan/frames_hi"
F_LO = "/tmp/refan/frames"

# 参考片：取前 12 张（0.5fps → 每 2s 一张，覆盖前 24s）
ref_files = sorted(os.listdir(F_HI))[:12]
ref = [os.path.join(F_HI, f) for f in ref_files]

# 我们成片的抽帧（若有）
ours_dir = "/ssd2/duanju/drama_1min/qc_final_v2"
ours = []
if os.path.isdir(ours_dir):
    fs = sorted(os.listdir(ours_dir))
    ours = [os.path.join(ours_dir, f) for f in fs if f.endswith(".jpg") or f.endswith(".png")][:12]

CW, CH = 240, 427
COLS = 6


def sheet(paths, title, out):
    rows = (len(paths) + COLS - 1) // COLS
    W = COLS * CW + (COLS + 1) * 8
    H = rows * (CH + 26) + 44
    canvas = Image.new("RGB", (W, H), (24, 24, 28))
    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    d.text((10, 10), title, fill=(240, 240, 240), font=font)
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("RGB")
        except Exception:
            continue
        im.thumbnail((CW, CH))
        r, c = divmod(i, COLS)
        x = 8 + c * (CW + 8)
        y = 36 + r * (CH + 26)
        canvas.paste(im, (x, y))
        d.rectangle([x, y, x + CW - 1, y + CH - 1], outline=(90, 90, 100))
        d.text((x + 2, y + CH + 4), os.path.basename(p)[:26], fill=(180, 180, 190), font=font)
    canvas.save(out, quality=88)
    print("wrote", out, canvas.size)


sheet(ref, "REF (360x640, 0.5fps first 24s)", "/tmp/refan/sheet_ref.jpg")
if ours:
    sheet(ours, "OURS v2 final", "/tmp/refan/sheet_ours.jpg")
else:
    print("no ours frames at", ours_dir)
