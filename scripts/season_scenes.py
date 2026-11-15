#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""季节与 24 节气场景词库。

每个变体含：
  key      变体标识（用于文件名）
  zh       中文名
  en       英文场景描述（用于生成）
  mood     氛围关键词（用于提示词与音效）
  clothes  该节气对应的着装/道具调整建议（叠加在角色原本服装之上）

冬/夏各一条 + 24 节气 = 26 个变体。
"""

# key, 中文名, 英文场景, 氛围关键词, 着装调整
VARIANTS = [
    # ---------- 冬 / 夏 ----------
    ("winter", "冬",
     "in deep winter, bare frost-covered trees and a cold overcast sky, breath faintly visible in the freezing air, "
     "snow dusted on the shoulders and hair, muted blue-grey light",
     "cold winter, freezing, muted blue grey",
     "wearing a heavy winter coat or padded jacket, a scarf and gloves, frost on the fabric"),
    ("summer", "夏",
     "in high summer, blazing bright sun and a hazy heat horizon, intense sunlight with hard shadows, "
     "sweat on the skin and collar, a faint shimmer of hot air in the distance",
     "hot summer, blazing sunlight, hazy heat",
     "in light summer clothing with the sleeves rolled up, collar open, sweat glistening on the skin"),

    # ---------- 24 节气 ----------
    ("lichun", "立春",
     "at the beginning of spring, the last thin snow melting into wet earth, tender green buds just breaking on the branches, "
     "soft pale morning light with a chill still in the air",
     "early spring thaw, pale light, first buds",
     "wearing a light spring jacket, the first buds and damp earth around"),
    ("yushui", "雨水",
     "during the rain of spring, thin silver rain falling steadily, wet reflective stone ground, "
     "damp haze in the air, water droplets on the hair and shoulders, the sky a soft overcast grey",
     "spring rain, damp haze, wet grey",
     "in a raincoat or with a translucent umbrella, wet shoulders and hair"),
    ("jingzhe", "惊蛰",
     "at the waking of insects in early spring, the first thunder rumbling in a still grey sky, "
     "the ground moist and soft, the first wild shoots pushing up through the soil",
     "first thunder, awakening earth, moist spring",
     "in a light jacket, standing on damp earth with new shoots around"),
    ("chunfen", "春分",
     "at the spring equinox, perfectly balanced warm daylight, fresh green fields stretching to the horizon, "
     "a bright azure sky with soft white clouds, gentle breeze moving the grass",
     "balanced spring daylight, fresh green, gentle breeze",
     "in light spring clothing, standing among fresh green fields"),
    ("qingming", "清明",
     "during the Qingming festival, soft misty rain over a hillside of blooming cherry and peach blossoms, "
     "petals drifting through the damp air, pale diffused light through the blossoms",
     "misty spring blossoms, drifting petals, pale diffused light",
     "in a light coat with an umbrella, surrounded by drifting blossom petals"),
    ("guyu", "谷雨",
     "at the end of the spring rains, lush green rice paddies filled with clear water reflecting the sky, "
     "warm humid air, thick growth everywhere, gentle rain-washed sunlight",
     "lush green paddy, humid warmth, rain-washed light",
     "in light clothing, sleeves rolled, standing among lush green paddies"),
    ("lixia", "立夏",
     "at the beginning of summer, the sun growing strong and warm, fresh green summer foliage in full leaf, "
     "bright warm golden light, the sky clear and luminous",
     "early summer warmth, fresh foliage, golden light",
     "in light summer clothing, warm sunlight on the skin"),
    ("xiaoman", "小满",
     "at the grain full stage of early summer, ripening wheat heads heavy with golden grain, "
     "the air hot and heavy, warm afternoon light raking across the fields, insects humming",
     "heavy ripe wheat, humid heat, raking light",
     "in a light shirt with sleeves rolled up, standing in heavy wheat"),
    ("mangzhong", "芒种",
     "during the grain in ear period, farmers harvesting under a blazing white-hot sun, "
     "sweat-soaked clothes and straw hats, thick humid haze over the fields, intense glare",
     "blazing harvest sun, heavy sweat, humid glare",
     "wearing a sweat-soaked work shirt and a straw hat, farm harvesting tools"),
    ("xiazhi", "夏至",
     "at the summer solstice, the longest day, the sun riding very high in a white-hot sky, "
     "harsh overhead light and short deep shadows, the air shimmering with heat, brilliant blue zenith",
     "longest day, white-hot sun, shimmering heat",
     "in the lightest summer clothing, strong sunlight overhead"),
    ("xiaoshu", "小暑",
     "during the minor heat of summer, the oppressive humidity rising, thick storm clouds building on the horizon, "
     "a green landscape under a heavy grey-green sky, distant lightning flash",
     "oppressive humidity, building thunderheads, heavy green sky",
     "in light summer clothing, a storm building behind"),
    ("dashu", "大暑",
     "during the major heat of summer, fierce midday sun beating down, the ground shimmering, "
     "sweat pouring and intense heat haze distorting the background, almost white sky",
     "fierce blazing heat, heat haze distortion, white sky",
     "in thin summer clothing, heavy sweat and heat haze"),
    ("liqiu", "立秋",
     "at the beginning of autumn, the first hint of cooling in the air, a thin morning mist over a field, "
     "the light turning warm and golden, the very first yellow leaf on an otherwise green tree",
     "first cool hint, warm golden light, thin mist",
     "in a light autumn layer, the first yellow leaf in the air"),
    ("chushu", "处暑",
     "at the end of the summer heat, the oppressive humidity finally easing, clear bright air with better visibility, "
     "warm afternoon sunlight, dry grass beginning to pale at the tips",
     "humidity easing, clear air, warm dry light",
     "in light clothing, the air finally drying out"),
    ("bailu", "白露",
     "during the white dew period, delicate white dew drops clinging to grass blades and spider silk, "
     "cool clear morning light with long low shadows, a pale silver-blue atmosphere",
     "white dew, cool clear morning, silver-blue air",
     "in a light coat, dew drops sparkling on the grass"),
    ("qiufen", "秋分",
     "at the autumn equinox, a crisp clear day with balanced golden light, a sky washed to deep blue, "
     "fields turning golden and rice paddies gleaming, gentle autumn breeze",
     "crisp golden clarity, deep blue sky, gentle breeze",
     "in light autumn clothing, standing in golden fields"),
    ("hanlu", "寒露",
     "during the cold dew period, the dew turning cold and heavy, a low mist over a hillside of chrysanthemums, "
     "the light cool and low with the sun barely warm, an autumn chill clearly in the air",
     "cold heavy dew, chrysanthemums, low cool light",
     "in a warmer autumn coat, cold dew on the grass"),
    ("shuangjiang", "霜降",
     "at the frost descent, white frost crystals covering the grass and roofs, cold blue morning light, "
     "breath visible in the freezing air, the landscape sharp and clear under a pale winter sky",
     "frost crystals, freezing blue light, visible breath",
     "in a heavy winter coat, frost on the shoulders, breath visible"),
    ("lidong", "立冬",
     "at the beginning of winter, the biting north wind stripping the last leaves, bare branches against a low grey sky, "
     "cold harsh light, everything quiet and still",
     "bitting north wind, bare branches, low grey sky",
     "in a heavy padded winter coat, scarf wrapped tight"),
    ("xiaoxue", "小雪",
     "during the minor snow, fine light snowflakes falling steadily against a cold grey sky, "
     "a thin dusting of snow on the ground and on dark rooftops, muted soft light",
     "fine falling snow, cold grey, soft muted light",
     "in a padded winter coat, fine snow settling on the shoulders"),
    ("daxue", "大雪",
     "during the major snow, heavy thick snow falling with low visibility, deep snow covering everything, "
     "everything white and hushed, flat diffused light with no shadows",
     "heavy snow, deep white cover, flat diffused light",
     "in a thick winter down coat, deep snow around"),
    ("dongzhi", "冬至",
     "at the winter solstice, the shortest day, the sun barely skimming the horizon in deep orange and violet, "
     "long raking shadows across frost and snow, the coldest still air",
     "shortest day, deep orange violet sunset, long shadows",
     "in a heavy winter coat, long cold shadows"),
    ("xiaohan", "小寒",
     "during the minor cold, severe winter chill, thick ice on the branches and puddles frozen solid, "
     "a pale washed-out sky, breath thick in the freezing air, everything still and crystalline",
     "severe chill, thick ice, pale washed-out sky",
     "in a thick down coat, thick ice glinting nearby"),
    ("dahan", "大寒",
     "during the major cold, the coldest season at its peak, deep snow and rime ice on every surface, "
     "an iron-grey sky, the harshest cold light, frozen breath and the silence of deep winter",
     "peak cold, rime ice, iron-grey silence",
     "in a thick down coat with a fur-lined hood, heavy rime frost nearby"),
]

JIEQI_KEYS = [v[0] for v in VARIANTS]

if __name__ == '__main__':
    print(f'变体总数: {len(VARIANTS)}')
    for k, zh, en, mood, cl in VARIANTS:
        print(f'  {k:14s} {zh}')
    assert len(VARIANTS) == 26
    assert len(set(JIEQI_KEYS)) == 26
