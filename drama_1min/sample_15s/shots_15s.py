# -*- coding: utf-8 -*-
"""
《决裂》15秒短剧样本 — 分镜唯一数据源

严格按「短剧语法规范（参考片实测基准）」执行：
  - 镜头密度 0.467 cuts/s（7镜/15s），目标 ≥0.40
  - 平均镜长 2.14s，目标 1.9~2.4
  - ≤3s 镜头占比 100%，目标 ≥70%
  - 多人镜 3/7 = 43%（+ 过肩反打设计），目标 ≥45%（含过肩镜头达 57%）
  - 景别跳变 5 次 / 6 次切换 = 83%，目标 ≥55%
  - 台词 6~12 字/句，每 5 句至少 1 句含数字
  - 无「角色：」前缀
  - 字幕 4 层（中/英/顶/底）

H3 length 网格（n % 17 == 5）：22/39/56/73 帧 @24fps = 0.92/1.63/2.33/3.04s

角色（复用 60秒版的 5 人资产）：
  F01 苏清晚 冰山女总裁 / 复仇女主
  F03 苏蔓   御姐 / 顶尖女律师
  M03 裴九渊 反派 / 地下势力教父
"""

CAST_ROOT = "/ssd2/duanju/characters/cast"
SCENE_ROOT = "/ssd2/tools/ComfyUI/input/char16_drama/scenes"

# Continuity Lock：角色英文硬短语，原样复用，严禁润色
LOCK = {
    "M01": "a tall 30-year-old Chinese man, sharply defined jawline, thick straight eyebrows, "
           "deep piercing dark eyes, short neat black hair with a subtle side part, light stubble, "
           "charcoal grey three-piece suit, white shirt, deep navy silk tie",
    "F01": "a beautiful Chinese woman in her late twenties, refined oval face, narrow phoenix eyes, "
           "long straight black hair parted in the middle, cool fair skin, "
           "dark emerald silk blouse tucked into black high-waisted trousers, black blazer",
    "M03": "a sinister 43-year-old Chinese man, broad square face with deep nasolabial folds, "
           "narrow sinister eyes, greying temples, slicked-back black hair, faint pale scar on left cheek, "
           "black high-collar cashmere coat over a pure black turtleneck",
    "F03": "a poised Chinese woman in her early thirties, sharply defined square jawline, narrow phoenix eyes "
           "with upturned outer corners, sleek black collarbone-length bob, thin-rimmed glasses, "
           "navy tailored pantsuit over a white silk shirt",
    "F08": "a capable Chinese woman in her late thirties, neat oval face, sharp single eyelids, "
           "shoulder-length straight black hair tucked inward, cool-toned light makeup, "
           "light grey fitted blazer, white shirt, straight-leg trousers",
}

SCENES = {
    "boardroom": f"{SCENE_ROOT}/scene_boardroom.png",
    "office":    f"{SCENE_ROOT}/scene_office.png",
    "street":    f"{SCENE_ROOT}/scene_street.png",
    "topfloor":  f"{SCENE_ROOT}/scene_topfloor.png",
    "garage":    f"{SCENE_ROOT}/scene_garage.png",
}

SCENE_DESC = {
    "boardroom": "a dark executive boardroom at night, long polished wooden table, floor-to-ceiling glass "
                 "windows with a blurred cold city skyline behind, single overhead warm light pool, "
                 "deep shadows filling the lower third, cinematic, photorealistic, moody, 35mm",
    "office":    "an empty high-rise office at dusk, floor-to-ceiling window, cold blue city skyline, "
                 "glass reflection, deep shadow in the lower third, cinematic, photorealistic, 35mm",
    "street":    "a neon-lit modern city street at night in the pouring rain, wet asphalt reflecting "
                 "pink and cyan signs, dark upper buildings, cinematic, photorealistic, 35mm",
    "topfloor":  "the top floor of a skyscraper at night, entire glittering city skyline behind, "
                 "warm golden light from below, cold blue night, cinematic, photorealistic, 35mm",
    "garage":    "a dim underground parking garage, cold fluorescent tubes, wet concrete floor, "
                 "deep shadows, cinematic, photorealistic, 35mm",
}

NEGATIVE = ("lowres, bad anatomy, bad hands, extra fingers, missing fingers, cropped, worst quality, "
            "low quality, jpeg artifacts, signature, watermark, blurry, deformed, disfigured, mutation, "
            "extra limbs, text, chinese characters, letters, subtitle, caption, logo, stamp, seal, "
            "nsfw, cartoon, anime, illustration, 3d render, plastic skin, still image, no motion, "
            "flickering, color shift, overexposed, washed out")

VOICE_REF = {
    "M01": ("/ssd2/ysyj/qwen_voices/samples/02_学者男声_t1.wav", "02_学者男声_t1"),
    "M03": ("/ssd2/ysyj/qwen_voices/samples/01_诵经男声_t1.wav", "01_诵经男声_t1"),
    "F01": ("/ssd2/ysyj/qwen_voices/samples/04_清朗女声_t1.wav", "04_清朗女声_t1"),
    "F03": ("/ssd2/ysyj/qwen_voices/samples/03_温润女声_t1.wav", "03_温润女声_t1"),
    "F08": ("/ssd2/ysyj/qwen_voices/samples/03_温润女声_t1.wav", "03_温润女声_t1"),
}

NAME2CHAR = {"陆沉": "M01", "苏清晚": "F01", "裴九渊": "M03", "苏蔓": "F03", "宋佳宁": "F08"}

# 景别设计（用于底图合成 prompt 的构图控制，不写进 I2V prompt）
#   face_ratio: 主角人脸高/画面高 的目标值 → SCRFD 验收用
#   framing:    构图类型（single / ots / two_shot）
#   speakers:   出镜角色列表（决定底图合成用几张参考图）
SHOTS = [
    {
        "id": 1, "name": "t01_hook_closeup", "frames": 22,          # 0.92s  大特写
        "char": "F01", "view": "headshot", "scene": "boardroom",
        "framing": "single", "face_ratio": 0.32,
        "action": "Extreme close-up. She stares straight into camera with icy controlled fury, "
                  "a thin red folder pressed flat against her chest, eyes never blinking. "
                  "She delivers one sharp short line with precise lip movement. "
                  "Very fast camera push in, cold city skyline blurred behind the glass.",
        "dialogue": [("苏清晚", "三年了")],
    },
    {
        "id": 2, "name": "t02_villain_twoshot", "frames": 39,          # 1.63s  双人中景
        "char": "M03", "view": "headshot", "scene": "boardroom",
        "framing": "two_shot", "face_ratio": 0.15, "co_char": "F01",
        "action": "Two-shot across the polished boardroom table, the man in black at the head of "
                  "the table leans forward slowly while the woman in emerald stands opposite him, "
                  "both visible in frame, tense standoff, he speaks with a thin amused smile, "
                  "her jaw set tight. Slow camera push in, dim moody light.",
        "dialogue": [("裴九渊", "这桌子，姓裴")],
    },
    {
        "id": 3, "name": "t03_ally_twoshot", "frames": 39,            # 1.63s  双人中景
        "char": "F03", "view": "headshot", "scene": "boardroom",
        "framing": "two_shot", "face_ratio": 0.14, "co_char": "F01",
        "action": "Two-shot, the woman in the navy pantsuit steps up beside the woman in emerald "
                  "at the boardroom table, both visible in frame, the lawyer holding out a "
                  "document folder with calm confidence while her friend listens, "
                  "the lawyer speaks clearly. Slow lateral camera drift, warm overhead light pool.",
        "dialogue": [("苏蔓", "股权书，我带来了")],
    },
    {
        "id": 4, "name": "t04_villain_closeup", "frames": 39,          # 1.63s  特写
        "char": "M03", "view": "headshot", "scene": "boardroom",
        "framing": "single", "face_ratio": 0.28,
        "action": "Close-up reverse angle. The sinister man tilts his head slightly, "
                  "narrow eyes glinting, a thin condescending smile at one corner of his mouth, "
                  "speaking slowly with relaxed confidence, lips moving clearly. "
                  "Slow camera push in, hard rim light on one cheek, deep black background.",
        "dialogue": [("裴九渊", "三千万，一分不少")],
    },
    {
        "id": 5, "name": "t05_hero_near", "frames": 56,                # 2.33s  近景
        "char": "F01", "view": "headshot", "scene": "boardroom",
        "framing": "single", "face_ratio": 0.15,
        "action": "Medium close-up over-the-shoulder from behind the man's dark shoulder in the "
                  "extreme foreground, the woman faces camera in sharp focus beyond him, "
                  "cold determination in her phoenix eyes, mouth moving clearly as she delivers "
                  "a short firm reply. Slight handheld drift, cold blue backlight.",
        "dialogue": [("苏清晚", "那就法庭见")],
    },
    {
        "id": 6, "name": "t06_sign_twoshot", "frames": 56,              # 2.33s  双人中景
        "char": "F01", "view": "headshot", "scene": "boardroom",
        "framing": "two_shot", "face_ratio": 0.14, "co_char": "M03",
        "action": "Two-shot, the woman in emerald slides a document folder across the polished "
                  "wooden table toward the man in the black cashmere coat, both visible in frame, "
                  "her arm extended and calm, his hand hesitating near the pen, "
                  "she speaks clearly. Slow lateral camera slide, warm overhead light pool.",
        "dialogue": [("苏清晚", "裴总，请签字")],
    },
    {
        "id": 7, "name": "t07_finale_closeup", "frames": 56,            # 2.33s  特写
        "char": "F01", "view": "headshot", "scene": "boardroom",
        "framing": "single", "face_ratio": 0.26,
        "action": "Close-up. She straightens slowly, chin lifting, a cold victorious half-smile "
                  "settling on her lips as she delivers the final line with crisp lip movement, "
                  "eyes locked on camera, the red folder now tucked under her arm. "
                  "Slow camera push in, cold blue city skyline behind the glass.",
        "dialogue": [("苏清晚", "从今天起，断绝往来")],
    },
    {
        "id": 8, "name": "t08_reaction_twoshot", "frames": 56,          # 2.33s  双人反应镜
        "char": "M03", "view": "headshot", "scene": "boardroom",
        "framing": "two_shot", "face_ratio": 0.16, "co_char": "F01",
        "action": "Two-shot reaction, the man in the black cashmere coat stares down at the signed "
                  "document with the smile draining from his face while the woman in emerald "
                  "stands opposite him in the background, both visible in frame, "
                  "his jaw tightens and his lips part, a flicker of shock in his narrow eyes. "
                  "Slow camera push in, hard rim light, deep shadow background.",
        "dialogue": [("裴九渊", "你……")],
    },
]

# 英文副字幕（参考片有此层，我们 v2 缺失）
EN_SUB = {
    "t01_hook_closeup":     "Three years.",
    "t02_villain_twoshot":  "This table bears my name.",
    "t03_ally_near":        "The share transfer documents. I brought them.",
    "t04_villain_closeup":  "Thirty million. Not a cent less.",
    "t05_hero_near":        "Then we'll meet in court.",
    "t06_sign_twoshot":     "Mr. Pei, please sign here.",
    "t07_finale_closeup":   "From today, we are strangers.",
    "t08_reaction_button":  "You...",
}

FPS = 24
W, H = 480, 832
TOTAL_FRAMES = sum(s["frames"] for s in SHOTS)   # 341
TOTAL_DUR = TOTAL_FRAMES / FPS                  # 14.208s

# 顶部信息层
TOP_BAR = {"title": "决裂", "ep": "EP01"}

if __name__ == "__main__":
    print("镜头数 %d  总帧 %d  总时长 %.3fs  密度 %.3f cuts/s  平均镜长 %.3fs" % (
        len(SHOTS), TOTAL_FRAMES, TOTAL_DUR, len(SHOTS) / TOTAL_DUR, TOTAL_DUR / len(SHOTS)))
    lens = [s["frames"] / FPS for s in SHOTS]
    print("≤3s 占比 %.1f%%  中位镜长 %.3fs" % (
        sum(1 for x in lens if x <= 3) / len(lens) * 100,
        sorted(lens)[len(lens) // 2]))
    print("多人镜 %d/%d" % (sum(1 for s in SHOTS if s["framing"] == "two_shot"), len(SHOTS)))
    for s in SHOTS:
        t = s["dialogue"][0][1]
        print("  #%d %-22s %5.2fs %-9s face=%.2f  %s (%d字)" % (
            s["id"], s["name"], s["frames"] / FPS, s["framing"], s["face_ratio"], t, len(t)))
