#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""jp200_data.txt -> scenes_jp200.json  (v2)

v2 变更（按用户三点决策）：
  1. 各占一半：每个类别 50 竖版(9:16) + 50 横版(16:9)  -> 总计 100 竖 + 100 横
  2. 街道场景后期加「真实日文招牌」：为有店招/灯笼/门牌的街道场景写入
     sign = {text, style, box, vertical}，文案全为真实日文（由真实字体渲染，
     见 paste_text.py / Noto Sans JP）。
  3. 自然环境场景无招牌（自然里本就没有字）。
"""
import json

SRC = '/Users/baidu/WorkBuddy/H100/bg100/jp200_data.txt'
DST = '/Users/baidu/WorkBuddy/H100/bg100/scenes_jp200.json'

# ---- 直立招牌框（归一化 x0,y0,x1,y1）----
BOX_V = [0.42, 0.085, 0.58, 0.35]     # 竖排小招牌（挂在上部）
BOX_H = [0.28, 0.10, 0.72, 0.215]     # 横排招牌（横跨上部）
BOX_L = [0.30, 0.075, 0.70, 0.245]    # 大灯笼/大门牌（横排，稍大）

# ---- 街道场景 -> (真实日文文案, 样式, 竖排?, 框) ----
# 样式取自 paste_text.STYLES: wood/darkwood/neon/neon_warm/neon_pink/paper/plaque/lightbox/banner/metal/gold/stone
SIGN_MAP = {
    'jpstr02_gion_lane':        ('祇園',     'darkwood', True,  BOX_V),
    'jpstr04_kiyomizu_slope':   ('清水坂',   'wood',     True,  BOX_V),
    'jpstr05_ninenzaka':        ('二年坂',   'wood',     True,  BOX_V),
    'jpstr06_sannenzaka':       ('三年坂',   'wood',     True,  BOX_V),
    'jpstr08_machiya_street':   ('呉服',     'darkwood', True,  BOX_V),
    'jpstr09_shibuya':          ('渋谷',     'neon',     False, BOX_H),
    'jpstr10_shinjuku':         ('新宿',     'neon',     False, BOX_H),
    'jpstr11_golden_gai':       ('酒場',     'paper',    True,  BOX_V),
    'jpstr12_omoide_yokocho':   ('焼鳥',     'paper',    True,  BOX_V),
    'jpstr13_ginza':            ('銀座',     'lightbox', False, BOX_H),
    'jpstr14_akihabara':        ('電気街',   'neon',     False, BOX_H),
    'jpstr15_nakamise':         ('仲見世',   'wood',     False, BOX_H),
    'jpstr16_kaminarimon':      ('雷門',     'plaque',   False, BOX_L),
    'jpstr17_dotonbori':        ('道頓堀',   'neon_warm', False, BOX_H),
    'jpstr18_shinsaibashi':     ('心斎橋',   'lightbox', False, BOX_H),
    'jpstr19_tsutenkaku':       ('通天閣',   'neon_warm', False, BOX_H),
    'jpstr20_shinsekai':        ('新世界',   'neon_warm', False, BOX_H),
    'jpstr22_susukino':         ('薄野',     'neon',     False, BOX_H),
    'jpstr24_convenience':      ('商店',     'lightbox', False, BOX_H),
    'jpstr25_izakaya_alley':    ('居酒屋',   'paper',    True,  BOX_V),
    'jpstr26_night_ramen':      ('拉麺',     'paper',    False, BOX_H),
    'jpstr28_karaoke_street':   ('歌謡',     'neon_pink', False, BOX_H),
    'jpstr29_shotengai':        ('商店街',   'wood',     False, BOX_H),
    'jpstr30_showa_retro':      ('昭和',     'wood',     False, BOX_H),
    'jpstr31_onsen_street':     ('温泉',     'paper',    True,  BOX_V),
    'jpstr32_onsen_yukata':     ('旅館',     'darkwood', True,  BOX_V),
    'jpstr33_seaside_village':  ('海鮮',     'wood',     False, BOX_H),
    'jpstr34_fishing_port':     ('漁港',     'wood',     False, BOX_H),
    'jpstr39_subway':           ('地下鉄',   'lightbox', False, BOX_H),
    'jpstr40_jr_station':       ('駅',       'lightbox', False, BOX_H),
    'jpstr41_shinkansen':       ('新幹線',   'lightbox', False, BOX_H),
    'jpstr44_station_plaza':    ('駅前',     'lightbox', False, BOX_H),
    'jpstr47_kyoto_tower':      ('京都',     'lightbox', False, BOX_H),
    'jpstr57_tea_house_street': ('茶屋',     'darkwood', True,  BOX_V),
    'jpstr58_kanazawa_higashi': ('茶屋街',   'darkwood', True,  BOX_V),
    'jpstr59_takayama_old':     ('酒蔵',     'darkwood', True,  BOX_V),
    'jpstr62_narai':            ('奈良井宿', 'wood',     True,  BOX_V),
    'jpstr69_nishiki':          ('錦市場',   'wood',     False, BOX_H),
    'jpstr70_pontocho':         ('先斗町',   'darkwood', True,  BOX_V),
    'jpstr84_arcade':           ('商店街',   'lightbox', False, BOX_H),
    'jpstr85_depachika':        ('食品',     'lightbox', False, BOX_H),
    'jpstr86_ramen_row':        ('拉麺',     'paper',    True,  BOX_V),
    'jpstr87_udon_shop':        ('うどん',   'paper',    False, BOX_H),
    'jpstr88_sushi_front':      ('寿司',     'wood',     False, BOX_H),
    'jpstr89_wagashi_shop':     ('和菓子',   'wood',     False, BOX_H),
    'jpstr90_kimono_rental':    ('呉服',     'wood',     True,  BOX_V),
    'jpstr91_temple_street':    ('土産',     'wood',     False, BOX_H),
    'jpstr92_shrine_market':    ('市',       'banner',   True,  BOX_V),
}

scenes = []
with open(SRC, encoding='utf-8') as f:
    for ln in f:
        ln = ln.rstrip('\n')
        if not ln.strip():
            continue
        parts = ln.split('\t')
        assert len(parts) == 4, f'bad line: {ln[:60]} ({len(parts)} fields)'
        sid, group, name, en = parts[0], parts[1], parts[2], parts[3]
        cat = 'nature' if sid.startswith('jpnat') else 'street'
        scenes.append({'sid': sid, 'cat': cat, 'group': group, 'name': name, 'en': en})

sids = [s['sid'] for s in scenes]
assert len(sids) == len(set(sids)), 'duplicate sid'

# ---- 各占一半：类内偶数 -> 竖版，奇数 -> 横版 ----
for cat in ('nature', 'street'):
    idx = 0
    for s in scenes:
        if s['cat'] != cat:
            continue
        s['orientation'] = 'portrait' if idx % 2 == 0 else 'landscape'
        idx += 1

# ---- 街道招牌 ----
for s in scenes:
    if s['sid'] in SIGN_MAP:
        text, style, vert, box = SIGN_MAP[s['sid']]
        s['sign'] = {'text': text, 'style': style, 'box': box, 'vertical': vert}

# ---- 统计 ----
n_p = sum(1 for s in scenes if s['orientation'] == 'portrait')
n_l = sum(1 for s in scenes if s['orientation'] == 'landscape')
nat = sum(1 for s in scenes if s['cat'] == 'nature')
str_ = sum(1 for s in scenes if s['cat'] == 'street')
n_sign = sum(1 for s in scenes if s.get('sign'))
assert n_p == 100 and n_l == 100, (n_p, n_l)

doc = {'meta': {'version': 'jp200-v2', 'count': len(scenes), 'nature': nat, 'street': str_,
                'portrait': n_p, 'landscape': n_l, 'with_sign': n_sign,
                'sizes': {'portrait': '832x1472 -> 9:16', 'landscape': '1472x832 -> 16:9'},
                'note': '日本氛围自然环境+街景；模型全程不产字；街道招牌真实日文后期合成(Noto Sans JP)'},
       'scenes': scenes}
with open(DST, 'w', encoding='utf-8') as f:
    json.dump(doc, f, ensure_ascii=False, indent=1)
print(f'wrote {DST}: total={len(scenes)} nature={nat} street={str_} '
      f'portrait={n_p} landscape={n_l} with_sign={n_sign}')
