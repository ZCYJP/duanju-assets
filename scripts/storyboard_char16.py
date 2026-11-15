# -*- coding: utf-8 -*-
"""
《深渊》3分钟都市短剧 — 陆沉 × 苏清晚 × 裴九渊
八镜三幕 ≈ 180s，全程对白无旁白。

每镜字段:
  scene_id / name / dur
  chars:  出场角色（决定用哪个视图做参考图）
  view:   该镜使用的角色参考图（hero / headshot / fullbody / threequarter ...）
  loc:    场景（决定场景参考图）
  action: 给 H3 的动作/表演描述（英文，Picture 1=人物 Picture 2=场景）
  dialogue: 中文台词，用 | 分隔每句，格式「说话人：台词」
  arc:    剧情功能

参考图目录: /ssd2/tools/ComfyUI/input/char16_drama/
"""

REF = {
    # 人物
    "M01": "char16_drama/M01/M01_hero.png",
    "M01_head": "char16_drama/M01/M01_headshot.png",
    "M01_body": "char16_drama/M01/M01_fullbody.png",
    "M01_angry": "char16_drama/M01/M01_angry.png",
    "M01_3q": "char16_drama/M01/M01_threequarter.png",
    "F01": "char16_drama/F01/F01_hero.png",
    "F01_head": "char16_drama/F01/F01_headshot.png",
    "F01_body": "char16_drama/F01/F01_fullbody.png",
    "F01_sad": "char16_drama/F01/F01_sad.png",
    "F01_3q": "char16_drama/F01/F01_threequarter.png",
    "M03": "char16_drama/M03/M03_hero.png",
    "M03_head": "char16_drama/M03/M03_headshot.png",
    "M03_body": "char16_drama/M03/M03_fullbody.png",
    "M03_angry": "char16_drama/M03/M03_angry.png",
    # 场景
    "loc_boardroom": "char16_drama/scenes/scene_boardroom.png",
    "loc_office": "char16_drama/scenes/scene_office.png",
    "loc_garage": "char16_drama/scenes/scene_garage.png",
    "loc_street": "char16_drama/scenes/scene_street.png",
    "loc_topfloor": "char16_drama/scenes/scene_topfloor.png",
}

SCENES = [
    # ========== 第一幕 · 决裂 ==========
    {
        "scene_id": 1, "name": "决裂", "dur": 23.0, "nsegs": 5,
        "chars": ["F01"], "view": "F01_head", "loc": "loc_boardroom",
        "action": (
            "Extreme close-up of <Picture 1>, a cold beautiful Chinese woman in her late twenties in a dark "
            "emerald silk blouse and black blazer, standing in the executive boardroom of <Picture 2>. She stares "
            "directly at the camera with icy controlled fury, a thin red folder pressed against her chest, her "
            "chin lifting slightly as she delivers a final, cutting line. Her lips move clearly and precisely with "
            "every word, her eyes never blinking, cold city skyline blurred through the glass behind her."
        ),
        "dialogue": (
            "苏清晚：陆沉，你我之间的账，今天结清。|"
            "苏清晚：从今天起，苏氏与陆氏，一刀两断。"
        ),
        "arc": "开场立冲突:女主主动切割,把男主逼到绝境"
    },
    {
        "scene_id": 2, "name": "来者不善", "dur": 23.0, "nsegs": 5,
        "chars": ["M03"], "view": "M03_head", "loc": "loc_boardroom",
        "action": (
            "Extreme close-up of <Picture 1>, a sinister 43-year-old Chinese man in a black high-collar cashmere "
            "coat, seated at the head of the boardroom table of <Picture 2>, cold narrow eyes and a faint pale scar "
            "on his left cheek. He leans forward slightly, a thin amused smile curling at one corner of his mouth, "
            "speaking slowly with the relaxed confidence of a man who already owns the room. His lips move clearly, "
            "dim moody lighting carving shadows across his face."
        ),
        "dialogue": (
            "裴九渊：苏小姐，董事会欠你一个交代。|"
            "裴九渊：可惜，从今天起，这张桌子姓裴了。"
        ),
        "arc": "反派登场,揭穿董事会已被裴九渊掌控"
    },
    {
        "scene_id": 3, "name": "深渊边缘", "dur": 23.0, "nsegs": 5,
        "chars": ["F01"], "view": "F01_sad", "loc": "loc_office",
        "action": (
            "Medium shot of <Picture 1>, the same Chinese woman in the emerald blouse and black blazer, standing "
            "alone by the floor-to-ceiling window of a high-rise office in <Picture 2>. She presses one hand flat "
            "against the glass, her reflection faint in the dark pane, shoulders lowered for the first time, eyes "
            "glistening with unshed tears. She speaks quietly to the empty room, her lips moving softly, cold dusk "
            "light outlining her silhouette against the city."
        ),
        "dialogue": (
            "苏清晚：三年……我一个人扛了三年。|"
            "苏清晚：爸，你走的时候说，会有人替我撑腰。"
        ),
        "arc": "女主独处卸下伪装,情绪谷底,埋下父亲旧案伏笔"
    },
    # ========== 第二幕 · 归来 ==========
    {
        "scene_id": 4, "name": "雨夜身影", "dur": 23.0, "nsegs": 5,
        "chars": ["M01"], "view": "M01_body", "loc": "loc_street",
        "action": (
            "Full body shot of <Picture 1>, a tall 30-year-old Chinese man in a charcoal three-piece suit and long "
            "dark wool coat, standing alone in the pouring rain on a neon-lit city street at night in <Picture 2>. "
            "Rainwater runs down his face and shoulders, his dark hair plastered flat, but his posture is "
            "immaculately straight and his gaze is fixed and unwavering. He takes one slow deliberate step forward "
            "through the rain and speaks with quiet absolute certainty, his lips moving clearly, neon reflections "
            "shimmering in the wet pavement around his shoes."
        ),
        "dialogue": (
            "陆沉：账不用你算。|"
            "陆沉：苏家的债，我三年前就接手了。"
        ),
        "arc": "男主最强钩子登场:三年前就已入局,悬念炸开"
    },
    {
        "scene_id": 5, "name": "亮底牌", "dur": 23.0, "nsegs": 5,
        "chars": ["M01"], "view": "M01_3q", "loc": "loc_street",
        "action": (
            "Three-quarter medium shot of <Picture 1>, the same man in the charcoal suit and dark coat, turning "
            "his head slightly toward the camera as he continues speaking in the rain of <Picture 2>. Rain streaks "
            "through the cold neon light, one side of his face lit sharp, the other in shadow, his expression "
            "quietly furious and absolutely controlled. He lifts his chin and delivers the final line with weight, "
            "his lips moving clearly and deliberately."
        ),
        "dialogue": (
            "陆沉：裴九渊动的是苏氏的股，不是苏氏的人。|"
            "陆沉：只要你还站在这间办公室，他就赢不了。"
        ),
        "arc": "男主亮出底牌:已布好反制,给女主希望"
    },
    {
        "scene_id": 6, "name": "正面交锋", "dur": 23.0, "nsegs": 5,
        "chars": ["M03", "M01"], "view": "M03_angry", "loc": "loc_boardroom",
        "action": (
            "Medium shot of <Picture 1>, the sinister man in the black high-collar coat, now standing and leaning "
            "over the boardroom table of <Picture 2>, his smile gone, cold eyes narrowed to slits, one hand pressing "
            "flat on the polished table as he speaks with cold fury. His lips move sharply, every word bitten off, "
            "hard overhead light throwing his shadow across the table, the moody boardroom looming behind him."
        ),
        "dialogue": (
            "裴九渊：陆沉，你还敢回来？|"
            "裴九渊：三年前没烧死你，是我的失误。"
        ),
        "arc": "反派与男主正面对撞,三年前有过一次交锋(旧案线索)"
    },
    # ========== 第三幕 · 反转与收束 ==========
    {
        "scene_id": 7, "name": "反转", "dur": 22.0, "nsegs": 3,
        "chars": ["F01"], "view": "F01_3q", "loc": "loc_topfloor",
        "action": (
            "Three-quarter medium shot of <Picture 1>, the Chinese woman in the emerald blouse and black blazer, "
            "standing tall on the top floor of a skyscraper at night in <Picture 2>, the entire city glittering "
            "behind her. She raises her chin, a faint confident smile crossing her face for the first time, and "
            "speaks with quiet triumph, her lips moving clearly, warm golden light from the boardroom table below "
            "rimming her shoulders against the cold blue night."
        ),
        "dialogue": (
            "苏清晚：裴先生，你买的是苏氏三年前的旧账。|"
            "苏清晚：可那份旧账，三年前就被人换成了你的。"
        ),
        "arc": "女主反杀:原来她早已配合陆沉做局,裴九渊才是猎物"
    },
    {
        "scene_id": 8, "name": "收束", "dur": 20.0, "nsegs": 2,
        "chars": ["M01", "F01"], "view": "M01", "loc": "loc_topfloor",
        "action": (
            "Cinematic medium two-subject framing: <Picture 1>, the man in the charcoal three-piece suit, stands "
            "on the top floor of <Picture 2> facing the camera with a rare quiet softness in his eyes, the woman in "
            "the emerald blouse visible in the foreground with her back to the camera, city lights glittering "
            "beyond the glass. He looks at her, then back toward the camera, and speaks the last line quietly, his "
            "lips moving clearly, warm gold and cold blue light meeting across the frame, wind moving her hair "
            "slightly. The image holds a beat after he finishes, an ending."
        ),
        "dialogue": (
            "陆沉：三年了，苏清晚。|"
            "陆沉：这一次，我站在你这边。"
        ),
        "arc": "情感收束+悬念留钩:三年前的约定指向未解旧案"
    },
]

if __name__ == "__main__":
    total = sum(s["dur"] for s in SCENES)
    print(f"八镜 总时长 {total:.0f}s ({total/60:.2f}min)")
    for s in SCENES:
        print(f'  Sc{s["scene_id"]} {s["name"]:6s} {s["dur"]:5.1f}s [{s["chars"][0]}@{s["view"]} / {s["loc"]}]')
