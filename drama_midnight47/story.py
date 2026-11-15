# -*- coding: utf-8 -*-
"""
《零点四十七单》— 分镜唯一数据源

规则（承自 drama_1min 已验证管线）：
  - 台词 ≤ 22 字（字幕单行上限），每镜 1 句主台词
  - I2V prompt 外观交给首帧，只写运镜/微动态/光影
  - 所有 prompt 末尾挂 no text / no subtitles
  - 场景图为 1248x832 横版，需先转 832x1248 竖版再合成

场景分工（第二阶段样本选取）：
  样本 A = S01 第四十七单（钩子：雨夜送单，1人）
  样本 B = S02 雨棚下（核心：叔侄交接日记本，2人）
"""
import os

BASE = "/ssd2/duanju/drama_midnight47"
CAST_ROOT = "/ssd2/duanju/characters/cast"
SCENE_ROOT = "/ssd2/duanju/scenes_100"
VOICE_ROOT = "/ssd2/duanju/voices"

# ---- 角色锚点：英文硬短语，下游原样复用，严禁 LLM 润色 ----
LOCK = {
    "M46": "a 23-year-old Chinese food delivery rider with a lean sun-tanned face, messy fringe, "
           "hurried eyes, wearing a yellow delivery uniform and a helmet",
    "F25": "an 18-year-old Chinese high school girl with a clean heart-shaped face, bright expressive eyes, "
           "straight bangs and black hair in a low ponytail, small tiger teeth, "
           "wearing a light blue school uniform jacket over a white top and pleated skirt with a shoulder bag",
    "M52": "a 56-year-old Chinese residential security guard with a round face, coarse skin, friendly eyes, "
           "buzz cut, wearing a dark blue security uniform with a shoulder radio",
    "F30": "a 63-year-old Chinese flower market grandmother with a deeply wrinkled round face, "
           "white hair in a bun, gentle eyes, wearing a faded blue blouse and apron, holding a bouquet",
    "F43": "a 59-year-old Chinese neighbourhood committee director with a round face, warm approachable eyes, "
           "deep wrinkles, hair pinned up, wearing a jujube-red jacket and a red volunteer vest",
    "M76": "a 61-year-old Chinese retired community party member with a round deeply lined face, kind eyes, "
           "neatly combed white hair, wearing a dark grey Zhongshan suit with a party badge",
    "M20": "a 55-year-old Chinese real estate tycoon, stocky round face, heavy jaw, shrewd laughing eyes, "
           "thinning black hair slicked shiny, a thick gold chain, wearing a deep burgundy velvet blazer "
           "over a black turtleneck with a jade ring",
    "M78": "a 35-year-old Chinese meat processing plant owner with a dark reddish face, thick brows, "
           "small lively eyes, a shaved head, thick neck, wearing a dark blue sanitised work suit and rubber gloves",
    "M60": "a 38-year-old Chinese ride-hailing driver with a square face, calm steady eyes, dark skin, "
           "shaved head, clean-shaven, wearing a black windbreaker over a white t-shirt",
    "F16": "a 33-year-old Chinese investigative journalist, lean angular face, sharp interrogative eyes, "
           "short black hair tucked behind her ears with faint shadows under her eyes, "
           "wearing a deep khaki trench coat over a black turtleneck",
}

# ---- 配音角色 → 专属基线音（绝对路径，一角色一音，绝不共用） ----
VOICE_REF = {
    "M46": f"{VOICE_ROOT}/samples_ext/M46_王大川_voice.wav",
    "F25": f"{VOICE_ROOT}/samples/F25_邱以沫_voice.wav",
    "F30": f"{VOICE_ROOT}/samples_ext/F30_刘桂香_voice.wav",
    "M52": f"{VOICE_ROOT}/samples_ext/M52_王保安_voice.wav",
    "F43": f"{VOICE_ROOT}/samples_ext/F43_黄秀英_voice.wav",
    "M76": f"{VOICE_ROOT}/samples_ext/M76_田建国_voice.wav",
    "M60": f"{VOICE_ROOT}/samples_ext/M60_孙强_voice.wav",
    "M20": f"{VOICE_ROOT}/samples/M20_何重山_voice.wav",
    "M78": f"{VOICE_ROOT}/samples_ext/M78_雷大勇_voice.wav",
    "F16": f"{VOICE_ROOT}/samples/F16_陈薇_voice.wav",
}

NAME2CHAR = {
    "王大川": "M46", "邱以沫": "F25", "刘桂香": "F30", "王保安": "M52",
    "黄秀英": "F43", "田建国": "M76", "孙强": "M60", "何重山": "M20",
    "雷大勇": "M78", "陈薇": "F16",
}

FEMALE = {"F16", "F25", "F30", "F43"}

# ---- 场景：横版原图 → 竖版目标 ----
SCENES = {
    "delivery_street": f"{SCENE_ROOT}/bg_delivery_street.png",
    "old_stairwell":   f"{SCENE_ROOT}/bg_old_stairwell.png",
    "canteen":         f"{SCENE_ROOT}/bg_canteen.png",
    "guardhouse":      f"{SCENE_ROOT}/bg_guardhouse.png",
    "wet_market":      f"{SCENE_ROOT}/bg_wet_market.png",
    "parking":         f"{SCENE_ROOT}/bg_cargo_deck.png",
    "office":          f"{SCENE_ROOT}/bg_open_office.png",
    "community":       f"{SCENE_ROOT}/bg_support_centre.png",
    "market_alley":    f"{SCENE_ROOT}/bg_market_alley.png",
    "luxury_lobby":    f"{SCENE_ROOT}/bg_corporate_lobby.png",
    "construction":    f"{SCENE_ROOT}/bg_construction_site.png",
    "hospital":        f"{SCENE_ROOT}/bg_hospital_dir.png",
    "flower_stall":    f"{SCENE_ROOT}/bg_flower_stall.png",
    "dorm_room":       f"{SCENE_ROOT}/bg_dorm_room.png",
}

SCENE_DESC = {
    "delivery_street": "a rain-soaked narrow Chinese city street at night, wet asphalt reflecting neon signs "
                       "in pink and cyan, low old residential buildings on both sides, tangled overhead cables, "
                       "heavy rainfall, cinematic, photorealistic, moody, 35mm, shallow depth of field",
    "old_stairwell":   "a derelict Chinese apartment building interior, bare concrete stairwell, chipped grey "
                       "walls with faded chalk numbers, dust in the air, a broken window letting in cold blue "
                       "night light, debris on the steps, cinematic, photorealistic, 35mm",
    "canteen":         "a small late-night Chinese noodle canteen, plastic tables and chairs, fluorescent tubes "
                       "hanging overhead, a wall calendar, greasy menu board, warm yellow light, "
                       "cinematic, photorealistic, 35mm",
    "guardhouse":      "a Chinese residential compound guardhouse at night, iron railing gate, a small booth "
                       "with a desk lamp glowing warm yellow, a CCTV monitor and an old wall clock inside, "
                       "wet pavement outside, cinematic, photorealistic, 35mm",
    "wet_market":      "a busy Chinese wet market in the early morning, plastic buckets full of flowers, "
                       "a concrete counter, hanging fluorescent lights, shoppers blurred in the background, "
                       "cinematic, photorealistic, 35mm",
    "parking":         "a dim concrete underground parking garage, harsh cold fluorescent tubes, "
                       "rows of parked cars, thick concrete pillars, wet stained floor, deep shadows, "
                       "cinematic, photorealistic, 35mm",
    "office":          "a cluttered Chinese community office room during the day, metal filing cabinets, "
                       "a paper covered notice board, an old ceiling fan, daylight through a barred window, "
                       "cinematic, photorealistic, 35mm",
    "community":       "a Chinese community activity room during the day, a long wooden table, "
                       "calligraphy scrolls on the wall, enamel teacups, folding metal chairs, "
                       "soft daylight from a large window, cinematic, photorealistic, 35mm",
    "market_alley":    "a narrow dark Chinese urban back alley at night, tall grimy walls, air conditioning "
                       "units and dripping pipes, a single broken street lamp, wet puddles, "
                       "cinematic, photorealistic, moody, 35mm",
    "luxury_lobby":    "an upscale Chinese real estate sales centre VIP lounge, giant illuminated city "
                       "rendering on the wall, dark leather sofas, polished marble coffee table, "
                       "warm golden recessed lighting, cinematic, photorealistic, 35mm",
    "construction":    "a Chinese urban demolition site at night, a bulldozer, broken concrete walls, "
                       "exposed rebar, floodlight towers casting harsh light, dust and debris, "
                       "cinematic, photorealistic, 35mm",
    "hospital":        "a Chinese hospital corridor during the day, pale green walls, fluorescent lights, "
                       "rows of blue plastic chairs, a queue window at the end, clean and cold, "
                       "cinematic, photorealistic, 35mm",
    "flower_stall":    "an outdoor Chinese flower market stall in the morning, buckets of carnations and "
                       "chrysanthemums, an old electronic scale, a concrete counter, soft overcast daylight, "
                       "cinematic, photorealistic, 35mm",
    "dorm_room":       "a small rented Chinese room at night, a narrow single bed, an old wooden table, "
                       "a bare bulb, a window with a mosquito net, worn walls, cinematic, photorealistic, 35mm",
}

NEGATIVE = ("lowres, bad anatomy, bad hands, extra fingers, missing fingers, cropped, worst quality, "
            "low quality, jpeg artifacts, signature, watermark, blurry, deformed, disfigured, mutation, "
            "extra limbs, text, chinese characters, letters, subtitle, caption, logo, stamp, seal, "
            "nsfw, cartoon, anime, illustration, 3d render, plastic skin, still image, no motion, "
            "flickering, color shift, overexposed, washed out, duplicated face, extra head")

# ---- 镜头表 ----
# id / name / dur(帧) / char / view / scene / action(I2V) / dialogue[(说话人, 台词)]
# 第二阶段样本 = 样本A(S01 全部 4 镜) + 样本B(S02 全部 5 镜)

SHOTS = [
    # ================= 样本 A：S01 第四十七单（钩子） =================
    {
        "id": 1, "name": "s01_a01_night_ride", "dur": 80, "sample": "A",
        "char": "M46", "view": "hero", "scene": "delivery_street",
        "action": "Medium shot from the side-front, riding an electric scooter through heavy rain on a narrow "
                  "city street at night. Rain streaks across the lens, neon signs smear into reflections on the "
                  "wet asphalt. His yellow helmet and jacket glow under the pink and cyan signage. He keeps "
                  "glancing down at the phone mounted on the handlebar. His lips move clearly as he speaks "
                  "into the shoulder radio. The scooter moves steadily forward, water spraying from the wheels. "
                  "Camera tracks alongside with slight handheld shake.",
        "dialogue": [("王大川", "四十七单，送完今天收工。")],
    },
    {
        "id": 2, "name": "s01_a02_phone_screen", "dur": 80, "sample": "A",
        "char": "M46", "view": "headshot", "scene": "delivery_street",
        "action": "Tight close-up of the man's face lit from below by a phone screen, rain running down his "
                  "helmet visor. He stops the scooter and stares down at the phone, his eyes narrowing and "
                  "his jaw tightening. His lips move clearly as he reads aloud. Rain hammers the awning above "
                  "him. The camera pushes in very slowly to an extreme close-up on his eyes, the phone glow "
                  "reflecting in them.",
        "dialogue": [("王大川", "槐树巷三号楼，三单元。")],
    },
    {
        "id": 3, "name": "s01_a03_demolished", "dur": 80, "sample": "A",
        "char": "M46", "view": "fullbody", "scene": "old_stairwell",
        "action": "Wide full body shot. The man stands in the cold blue rain inside a demolished building shell, "
                  "bare concrete and exposed rebar around him, a tilted hand-held flashlight throwing a hard "
                  "sweeping beam across rubble. He tilts his head up, looking at a faded chalk number on the "
                  "wall. His lips move clearly as he speaks. Water drips steadily from the broken ceiling. "
                  "The camera cranes slowly downward and holds.",
        "dialogue": [("王大川", "这楼，去年就拆了。")],
    },
    {
        "id": 4, "name": "s01_a04_final_call", "dur": 80, "sample": "A",
        "char": "M46", "view": "headshot", "scene": "delivery_street",
        "action": "Close-up, the man under a street lamp in the rain, head slightly bowed, the phone held low "
                  "in his hand with its screen still glowing. Rain runs down his face and he does not wipe it "
                  "away. His lips move slowly and clearly as he speaks into the radio, the words coming out "
                  "quietly. His eyes stay fixed on the phone. The camera holds almost still, only a very slow "
                  "creeping push in, the rain visible in the lamplight.",
        "dialogue": [("王大川", "妈，今晚不回去吃了。")],
    },

    # ================= 样本 B：S02 雨棚下（叔侄交接） =================
    {
        "id": 5, "name": "s02_b01_figure_under_umbrella", "dur": 80, "sample": "B",
        "char": "F25", "view": "fullbody", "scene": "old_stairwell",
        "action": "Full body shot of a young girl standing completely still under a black umbrella in the "
                  "rain, inside the skeleton of a demolished building, bare concrete columns and rebar behind "
                  "her. Her light blue school uniform jacket is already dark with rain at the shoulders. "
                  "Her face is calm and unreadable, eyes fixed forward. She does not move at all. "
                  "The camera slowly dollies forward through the rain toward her.",
        "dialogue": [],
    },
    {
        "id": 6, "name": "s02_b02_turn_and_face", "dur": 80, "sample": "B",
        "char": "F25", "view": "headshot", "scene": "old_stairwell",
        "action": "Medium close-up of the girl. She turns her head slightly toward the camera and her lips "
                  "move clearly as she speaks, one word only, her voice flat and controlled. Rain falls behind "
                  "her in the dark, the umbrella edge trembling slightly at the top of frame. "
                  "The camera holds steady with a very slow push in to a close-up.",
        "dialogue": [("邱以沫", "大川哥。")],
    },
    {
        "id": 7, "name": "s02_b03_give_notebook", "dur": 80, "sample": "B",
        "char": "F25", "view": "threequarter", "scene": "old_stairwell",
        "action": "Medium three-quarter shot. The girl reaches both hands forward and passes a worn dark "
                  "notebook out of frame to someone. Her fingers tighten on the notebook for a moment before "
                  "releasing it. Rain drips from the umbrella rim. Her lips move clearly as she speaks while "
                  "her hands extend. The camera pushes in slowly, framing her hands and her face together.",
        "dialogue": [("邱以沫", "这单是我下的。")],
    },
    {
        "id": 8, "name": "s02_b04_rider_reacts", "dur": 80, "sample": "B",
        "char": "M46", "view": "headshot", "scene": "old_stairwell",
        "action": "Close-up of the delivery rider, his yellow helmet still on, rainwater running down his face, "
                  "the phone screen light on his cheek. He looks at the notebook being handed to him, his eyes "
                  "widen slightly, his lips part. His lips move clearly as he speaks, the words barely audible. "
                  "He reaches forward with one hand. The camera holds on his face, a slow creeping push in.",
        "dialogue": [("王大川", "这是……大勇的？")],
    },
    {
        "id": 9, "name": "s02_b05_notebook_page", "dur": 80, "sample": "B",
        "char": "M46", "view": "threequarter", "scene": "old_stairwell",
        "action": "Over-the-shoulder close-up from behind the man, looking down at the open worn notebook in "
                  "his hands. His thumb rubs slowly across the page. Rain falls around him, his shoulders "
                  "darker with water. His lips move clearly as he reads the words on the page. "
                  "The camera drifts slowly down toward the page, then back up to his face.",
        "dialogue": [("王大川", "缺三个扣件，已报，未修。")],
    },
]

# ---- 样本分组 ----
SAMPLES = {
    "A": {
        "name": "样本A_S01第四十七单",
        "shots": [s["name"] for s in SHOTS if s["sample"] == "A"],
        "title": "零点四十七单",
        "subtitle": "第一集 · 钩子",
    },
    "B": {
        "name": "样本B_S02雨棚下",
        "shots": [s["name"] for s in SHOTS if s["sample"] == "B"],
        "title": "零点四十七单",
        "subtitle": "第一集 · 日记本",
    },
}

if __name__ == "__main__":
    print("shots:", len(SHOTS))
    for k, v in SAMPLES.items():
        print(k, v["name"], len(v["shots"]), "shots")
    # 一致性自检：参考音唯一性 + 台词长度
    # 一角色一参考音：检查 VOICE_REF 的「角色→参考音」映射是单射（不同角色不撞声）
    vals = list(VOICE_REF.values())
    print("voice refs unique per role:", len(set(vals)) == len(vals),
          len(set(vals)), "/", len(vals))
    # 出镜角色的参考音必须唯一
    on_screen = sorted({NAME2CHAR[sp] for sh in SHOTS for sp, _ in sh["dialogue"]})
    on_refs = [VOICE_REF[c] for c in on_screen]
    print("on-screen chars:", on_screen)
    print("on-screen refs unique:", len(set(on_refs)) == len(on_refs))
    bad = [(sh["name"], t) for sh in SHOTS for _, t in sh["dialogue"] if len(t) > 22]
    print("over-22-char lines:", bad if bad else "none")
    for sh in SHOTS:
        cid = sh["char"]
        assert os.path.exists(f"{CAST_ROOT}/{cid}/{cid}_{sh['view']}.png"), (cid, sh["view"])
        assert os.path.exists(SCENES[sh["scene"]]), sh["scene"]
    for cid, p in VOICE_REF.items():
        assert os.path.exists(p), (cid, p)
    print("asset paths: all OK")
