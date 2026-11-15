# -*- coding: utf-8 -*-
"""
《决裂》60秒都市商战悬疑短剧 — 分镜唯一数据源（h100-038 低分辨率版）

角色（5人，取自 /ssd2/duanju/characters/cast/ 150人库，每��10视图832x1248）：
  M01 陆沉   都市霸总 / 冷峻商业领袖
  F01 苏清晚 冰山女总裁 / 复仇女主
  M03 裴九渊 反派 / 地下势力教父
  F03 苏蔓   御姐 / 顶尖女律师
  F08 宋佳宁 职场精英 / 女高管

规格：480x832 竖版 / 16fps / 80帧每镜(5.0s) / 全对白无旁白 / 12镜总时长60.0s
每镜字段：
  id / name / dur(帧) / char / view / scene / action(英文,I2V留白) / dialogue
  dialogue: [(说话人, 台词), ...]  —— 台词<=15字

硬规：
  - 台词 <=15 字/句
  - I2V prompt 外观交给首帧，只写运镜/微动态/光影
  - 所有 prompt 末尾挂 no text / no subtitles
  - 底图先用「角色参考图 + 场景图」合成，再做 I2V
"""

CAST_ROOT = "/ssd2/duanju/characters/cast"
SCENE_ROOT = "/ssd2/tools/ComfyUI/input/char16_drama/scenes"

# Continuity Lock：角色英文硬短语，下游原样复用，严禁 LLM 润色
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

# 场景参考图（832x1248 竖版；topfloor 为 1392x752 横版，用 I2V 自动裁切）
SCENES = {
    "boardroom": f"{SCENE_ROOT}/scene_boardroom.png",
    "office":    f"{SCENE_ROOT}/scene_office.png",
    "street":    f"{SCENE_ROOT}/scene_street.png",
    "topfloor":  f"{SCENE_ROOT}/scene_topfloor.png",
    "garage":    f"{SCENE_ROOT}/scene_garage.png",
}

# 场景英文描述（供底图合成用）
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

# ---- 12镜 × 5秒 = 60.0秒 ----
# dur 帧数(49帧/镜=3.06s)，台词放在对应镜内
SHOTS = [
    {
        "id": 1, "name": "s01_hook_fold", "dur": 80,
        "char": "F01", "view": "headshot", "scene": "boardroom",
        "action": "Extreme close-up. She stares straight into camera with icy controlled fury, a thin "
                  "red folder pressed flat against her chest. She lifts her chin a fraction and speaks "
                  "with sharp precise lip movement, eyes never blinking. Slow camera push in, "
                  "cold city skyline blurred behind the glass.",
        "dialogue": [("苏清晚", "三年了。这笔账，今天结清。")],
    },
    {
        "id": 2, "name": "s02_ally_lawyer", "dur": 80,
        "char": "F03", "view": "headshot", "scene": "boardroom",
        "action": "Medium close-up of the same woman from a slight low angle. She leans forward a "
                  "little, brows raised with calm confidence, mouth moving clearly and deliberately as "
                  "she speaks. One hand rests on the table. Slow lateral camera drift, warm light "
                  "carving her face out of the dark.",
        "dialogue": [("苏蔓", "清晚，股权书我带来了。")],
    },
    {
        "id": 3, "name": "s03_villain_seat", "dur": 80,
        "char": "M03", "view": "headshot", "scene": "boardroom",
        "action": "Medium shot. The man sits at the head of the table, leans forward slowly, a thin "
                  "amused smile curling at one corner of his mouth, speaking slowly with relaxed "
                  "confidence. His lips move clearly. Slow camera push in, dim moody light, faint "
                  "smoke drifting through the background.",
        "dialogue": [("裴九渊", "苏小姐，这张桌子，姓裴了。")],
    },
    {
        "id": 4, "name": "s04_exec_silent", "dur": 80,
        "char": "F08", "view": "headshot", "scene": "boardroom",
        "action": "Medium close-up of the woman in the light grey blazer standing beside the table. "
                  "She holds a tablet against her chest, jaw set tight, eyes flicking sideways toward "
                  "the man at the head of the table. Her lips part but she says nothing. Slow camera "
                  "push in, cold overhead light.",
        "dialogue": [("宋佳宁", "……董事会，签到齐了。")],
    },
    {
        "id": 5, "name": "s05_break_alone", "dur": 80,
        "char": "F01", "view": "sad", "scene": "office",
        "action": "Medium shot of the woman standing alone by the floor-to-ceiling window, one hand "
                  "pressed flat against the cold glass, shoulders lowered for the first time, eyes "
                  "glistening with unshed tears. Her lips move softly as she speaks to the empty room. "
                  "Slow camera push in, cold dusk light outlining her silhouette against the city.",
        "dialogue": [("苏清晚", "爸，你走的那年，我十九。")],
    },
    {
        "id": 6, "name": "s06_garage_arrival", "dur": 80,
        "char": "M01", "view": "fullbody", "scene": "garage",
        "action": "Full body shot. The tall man stands alone in the dim underground garage, coat "
                  "unbuttoned, posture impeccably straight. He takes one slow deliberate step "
                  "forward, head slightly lowered, then raises his gaze. Slow tracking camera, cold "
                  "fluorescent light, wet concrete reflecting his shoes.",
        "dialogue": [("陆沉", "三年，我等这句话。")],
    },
    {
        "id": 7, "name": "s07_rain_street", "dur": 80,
        "char": "M01", "view": "hero", "scene": "street",
        "action": "Full body shot in the pouring rain on the neon-lit city street at night. Rainwater "
                  "runs down his face and shoulders, dark hair plastered flat, but his gaze is fixed and "
                  "unwavering. He speaks with quiet absolute certainty, lips moving clearly. Slow "
                  "camera push in, neon reflections shimmering in the wet pavement.",
        "dialogue": [("陆沉", "苏家的债，我三年前就接手了。")],
    },
    {
        "id": 8, "name": "s08_villain_rage", "dur": 80,
        "char": "M03", "view": "angry", "scene": "boardroom",
        "action": "Medium shot. The man has stood up and leans over the polished table, his smile "
                  "gone, eyes narrowed to slits, one hand pressed flat on the wood. He speaks sharply, "
                  "every word bitten off, jaw clenched. Slow camera push in, hard overhead light "
                  "throwing his shadow across the table.",
        "dialogue": [("裴九渊", "陆沉，你还敢回来？")],
    },
    {
        "id": 9, "name": "s09_lawyer_cards", "dur": 80,
        "char": "F03", "view": "threequarter", "scene": "boardroom",
        "action": "Three-quarter medium shot of the woman in the navy pantsuit. She lifts a thin "
                  "folder slightly, a faint knowing smile at the corner of her mouth, speaking "
                  "measured and calm with clear lip movement. Slow camera arc around her, warm light "
                  "on one side of her face, dark boardroom behind.",
        "dialogue": [("苏蔓", "裴先生，文件我念给你听。")],
    },
    {
        "id": 10, "name": "s10_exec_reveal", "dur": 80,
        "char": "F08", "view": "threequarter", "scene": "boardroom",
        "action": "Medium close-up of the woman in the light grey blazer. She turns her head slowly "
                  "from the man at the head of the table toward the camera, and a small controlled "
                  "smile appears. Her lips move clearly and deliberately. Slow camera push in, cold "
                  "overhead light, deep boardroom shadow.",
        "dialogue": [("宋佳宁", "签约方，写的是裴九渊。")],
    },
    {
        "id": 11, "name": "s11_topfloor_reversal", "dur": 80,
        "char": "F01", "view": "threequarter", "scene": "topfloor",
        "action": "Three-quarter medium shot of the woman standing tall on the top floor of the "
                  "skyscraper at night, entire glittering city behind her. She raises her chin and a "
                  "confident smile crosses her face for the first time, lips moving clearly with "
                  "quiet triumph. Slow camera dolly in, warm golden light rimming her shoulders "
                  "against the cold blue night.",
        "dialogue": [("苏清晚", "你买走的，是我的旧账。")],
    },
    {
        "id": 12, "name": "s12_finale_hook", "dur": 80,
        "char": "M01", "view": "threequarter", "scene": "topfloor",
        "action": "Three-quarter medium shot of the man, backlit by the glittering city skyline. He "
                  "turns his head slightly toward camera, a very faint smile at the corner of his mouth, "
                  "and speaks the final line with quiet weight, lips moving clearly and deliberately. "
                  "Slow camera push in, warm gold light from below against cold blue night.",
        "dialogue": [("陆沉", "苏清晚，欢迎回家。")],
    },
]

# 角色 -> 音色参考（Qwen3-TTS 声纹克隆）
# 男声用 01_诵经男声 / 02_学者男声；女声用 03_温润女声 / 04_清朗女声
VOICE_REF = {
    "M01": ("/ssd2/ysyj/qwen_voices/samples/02_学者男声_t1.wav", "02_学者男声_t1"),
    "M03": ("/ssd2/ysyj/qwen_voices/samples/01_诵经男声_t1.wav", "01_诵经男声_t1"),
    "F01": ("/ssd2/ysyj/qwen_voices/samples/04_清朗女声_t1.wav", "04_清朗女声_t1"),
    "F03": ("/ssd2/ysyj/qwen_voices/samples/03_温润女声_t1.wav", "03_温润女声_t1"),
    "F08": ("/ssd2/ysyj/qwen_voices/samples/03_温润女声_t1.wav", "03_温润女声_t1"),
}

# 说话人 -> 角色ID
NAME2CHAR = {"陆沉": "M01", "苏清晚": "F01", "裴九渊": "M03", "苏蔓": "F03", "宋佳宁": "F08"}

if __name__ == "__main__":
    total = 0
    print(f"镜头数: {len(SHOTS)}")
    for s in SHOTS:
        n = len(s["dialogue"])
        for sp, tx in s["dialogue"]:
            assert len(tx) <= 15, f"台词超15字: {sp} {tx}({len(tx)})"
            assert sp in NAME2CHAR, f"未知说话人 {sp}"
        total += s["dur"]
        print(f"  s{s['id']:02d} {s['name']:<20} {s['char']} {s['view']:<12} {s['scene']:<10} "
              f"台词{n}句")
    print(f"总帧数 {total} @16fps = {total/16:.2f}s")
    # 角色资产校验
    import os
    for c in set(NAME2CHAR.values()):
        for v in ["hero", "headshot", "fullbody", "threequarter", "sad", "angry"]:
            p = f"{CAST_ROOT}/{c}/{c}_{v}.png"
            assert os.path.exists(p), f"缺资产 {p}"
    for k, p in SCENES.items():
        assert os.path.exists(p), f"缺场景 {p}"
    print("资产校验通过")
