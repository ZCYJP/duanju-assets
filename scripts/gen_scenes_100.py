#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 100 张剧情背景图，场景内含真实汉字/繁体/英文/日文/韩文招牌。

要点：
  * 背景必须无人物（negative prompt 强制 people/mannequin 等）
  * 文字内容写进 prompt 的引号里，Qwen-Image 会按字形渲染
  * 每张背景用独立 prefix，避免多实例共享 output 目录时互相覆盖
  * 支持 --shard/--of 多实例并行
"""
import json, os, sys, time, argparse, urllib.request, urllib.parse

sys.path.insert(0, '/ssd2/short_drama/dh_scripts')
import comfy

OUT = "/ssd2/duanju/scenes_100"

STYLE = ("photorealistic cinematic environment shot, natural lighting, ultra detailed, 8k uhd, "
         "architectural photography, wide angle, sharp focus, no people, no human figures, "
         "no mannequin, no crowd, no silhouette of a person")

NEG = ("people, person, human, crowd, figures, faces, bodies, mannequin, silhouette of a person, "
       "lowres, bad anatomy, extra limbs, watermark, blurry, distorted text, garbled letters, "
       "cartoon, anime, illustration, painting")

# (scene_id, 中文描述, 英文描述, 招牌文字, 文字语言标签)
SCENES = [
 ("bg_factory_gate", "国营老厂正门，斑驳水泥墙与铁栅栏大门", "the front gate of a state-owned factory with weathered concrete walls and a wrought iron gate", "国营第三机械厂", "简体中文"),
 ("bg_steel_mill", "钢铁厂高炉车间，橙红炉火与蒸汽管道", "a steel mill blast furnace hall with orange furnace glow and steam pipes", "安全第一 预防为主", "简体中文"),
 ("bg_wheat_field", "华北麦田，金色麦浪延伸至地平线", "a wheat field in northern China with golden waves reaching the horizon", "丰收中国", "简体中文"),
 ("bg_fishing_dock", "渔港码头，渔船与堆叠渔网", "a fishing harbour dock with boats and stacked fishing nets", "漁港", "繁体中文"),
 ("bg_classroom", "中学教室，绿色黑板与成排课桌", "a middle school classroom with a green blackboard and rows of desks", "知识改变命运", "简体中文"),
 ("bg_dorm_room", "大学宿舍，桌面有代码书与显示器", "a university dorm room with code books and a monitor on the desk", "计算机学院", "简体中文"),
 ("bg_tile_warehouse", "建材市场仓库，堆叠瓷砖与板材", "a building materials warehouse stacked with tiles and boards", "建材批发中心", "简体中文"),
 ("bg_tcm_clinic", "中医馆内景，雕花木柜与药罐", "the interior of a traditional Chinese medicine clinic with carved cabinets", "同仁堂", "繁体中文"),
 ("bg_parcel_station", "小区快递驿站门口，成堆纸箱", "the entrance of a neighbourhood parcel station stacked with boxes", "快递代收点", "简体中文"),
 ("bg_fire_station", "消防站训练场，红色警灯与水带", "a fire station training ground with red lights and hoses", "人民消防 为民服务", "简体中文"),
 ("bg_track_field", "田径场跑道，起跑线与观众席", "an athletics track at the starting line with spectator stands", "更快 更高 更强 更团结", "简体中文"),
 ("bg_corporate_lobby", "高层办公大堂，玻璃幕墙与城市景观", "a high-rise corporate lobby with glass curtain walls and a city view", "远景集团", "简体中文"),
 ("bg_supermarket", "社区超市收银区，货架与购物车", "a community supermarket checkout area with shelves and carts", "惠民超市", "简体中文"),
 ("bg_law_office", "律师事务所，皮革书架与合同堆", "a law office with leather bookshelves and stacks of contracts", "诚信律师事務所", "繁体中文"),
 ("bg_school_office", "中学教师办公室，作业本与红笔堆叠", "a middle school office stacked with notebooks and red pens", "教研室", "简体中文"),
 ("bg_fresh_aisle", "超市生鲜区，冷柜与蔬菜堆", "the fresh produce section of a supermarket with fridges and vegetables", "新鲜蔬菜 每日直供", "简体中文"),
 ("bg_logistics_yard", "物流仓库门口，成排厢式货车", "a logistics warehouse yard lined with cargo trucks", "物流园", "简体中文"),
 ("bg_kindergarten", "幼儿园教室，彩色积木与矮桌椅", "a kindergarten classroom with colorful blocks and small tables", "快乐成长", "简体中文"),
 ("bg_wet_market", "菜市场通道，摊位与悬挂的灯泡", "a wet market aisle with stalls and hanging light bulbs", "菜市場", "繁体中文"),
 ("bg_delivery_street", "城市街道，电动车车流与店铺", "a city street with streams of electric scooters and shops", "外卖驿站", "简体中文"),
 ("bg_precision_factory", "精密制造车间，机械臂与工作台", "a precision manufacturing floor with robotic arms and workbenches", "智能制造", "简体中文"),
 ("bg_open_office", "开放式办公区，双屏电脑与绿植", "an open plan office with dual monitors and green plants", "创科网络科技", "简体中文"),
 ("bg_old_stairwell", "老小区楼道，窗边阳光与藤椅", "an old apartment stairwell with sunlight and a rattan chair", "幸福家园", "简体中文"),
 ("bg_game_studio", "游戏公司工位，双屏与手办", "a game studio desk with dual monitors and figurines", "像素工作室", "简体中文"),
 ("bg_guardhouse", "小区门岗亭，铁栅栏门与监控屏", "a residential guardhouse with a metal gate and monitor screen", "小区保安室", "简体中文"),
 ("bg_nail_salon", "美甲工作室，粉色装饰与灯箱", "a nail salon with pink decor and a lightbox sign", "指尖美学", "简体中文"),
 ("bg_driving_school", "驾校训练场，教练车与锥桶", "a driving school practice ground with a training car and cones", "宏达驾校", "简体中文"),
 ("bg_playground", "儿童游乐场，滑梯与彩色球", "a children's playground with a slide and colourful balls", "儿童乐园", "简体中文"),
 ("bg_reno_site", "装修工地，毛坯房与梯子", "a renovation site with an unfinished apartment and ladders", "精工装修", "简体中文"),
 ("bg_startup_desk", "创业公司工位，多个显示器与白板", "a startup desk with multiple monitors and a whiteboard", "云启科技", "简体中文"),
 ("bg_butchershop", "肉铺案台，挂着的新鲜猪肉", "a butcher counter with fresh cuts of meat hanging", "鲜肉铺", "简体中文"),
 ("bg_car_interior", "网约车驾驶座，仪表盘与夜景", "the driver seat of a ride-hailing car with a dashboard and night view", "安全驾驶", "简体中文"),
 ("bg_train_platform", "老火车站站台，铁轨与绿皮火车", "an old railway platform with tracks and a green train", "懷舊車站", "繁体中文"),
 ("bg_lumber_yard", "木材场，堆叠的圆木", "a lumber yard with stacked round logs", "木材加工", "简体中文"),
 ("bg_gym", "健身房器械区，杠铃与跑步机", "a gym equipment area with barbells and treadmills", "力量训练", "简体中文"),
 ("bg_hospital_dir", "县医院院长办公室，书柜与地球仪", "a hospital director's office with bookshelves and a globe", "人民医院", "简体中文"),
 ("bg_art_studio", "美院画室，画架与油画", "an art studio with easels and oil paintings", "美術學院", "繁体中文"),
 ("bg_dispatch_room", "机械厂调度室，白板与排产表", "a factory scheduling office with a whiteboard and production schedule", "生产调度室", "简体中文"),
 ("bg_textile_mill", "纺织车间，整齐排列的纺机", "a textile mill with rows of spinning machines", "纺织车间", "简体中文"),
 ("bg_delivery_station", "外卖站点，电动车整齐停放", "a delivery station with electric scooters parked in rows", "蜂鸟配送", "简体中文"),
 ("bg_boutique", "极简品牌展厅，白墙与产品台", "a minimalist brand showroom with white walls and product pedestals", "精品店", "简体中文"),
 ("bg_sample_room", "服装厂样衣间，人台与布料卷", "a garment factory sample room with mannequins and fabric rolls", "样衣间", "简体中文"),
 ("bg_principal_office", "幼儿园园长办公室，墙贴满幼儿画", "a kindergarten principal's office covered with children's art", "幼稚園", "日语"),
 ("bg_back_office", "超市仓库办公室，成排货架与文件", "a supermarket back office with shelving and paperwork", "採購部", "繁体中文"),
 ("bg_flower_stall", "花市小摊，鲜花与塑料桶", "a flower stall with buckets of fresh blooms", "鮮花", "繁体中文"),
 ("bg_yoga_studio", "瑜伽馆落地窗前，木地板与绿植", "the floor-to-ceiling windows of a yoga studio with wooden floors", "瑜伽馆", "简体中文"),
 ("bg_nursery", "月子中心婴儿房，成排婴儿床", "a postnatal care centre nursery with rows of bassinets", "母婴护理中心", "简体中文"),
 ("bg_airport", "机场候机厅，落地窗与航班屏", "an airport departure lounge with windows and flight boards", "出发大厅", "简体中文"),
 ("bg_cafe", "写字楼咖啡厅，洽谈的沙发区", "an office building cafe with a negotiation seating area", "COFFEE", "英语"),
 ("bg_makeup_room", "化妆间，化妆镜灯环与刷具", "a makeup room with a lit mirror and brushes", "Beauty Studio", "英语"),
 ("bg_home_services", "家政公司办公室，墙上贴着服务项目表", "a home services office with a service list on the wall", "家政服務", "繁体中文"),
 ("bg_kindergarten_paint", "幼儿园活动室，玩具墙与地毯", "a kindergarten activity room with a toy wall and carpets", "あそび", "日语"),
 ("bg_bubble_tea", "奶茶店吧台，饮品菜单灯箱", "a bubble tea counter with an illuminated menu board", "奶茶の店", "日语"),
 ("bg_beauty_room", "美容院护理间，美容床与绿植", "a beauty treatment room with a treatment bed and plants", "Beauty Care", "英语"),
 ("bg_industrial_office", "工业园区办公楼，落地窗与厂房天际线", "an industrial park office with floor-to-ceiling windows and a factory skyline", "工业制造基地", "简体中文"),
 ("bg_tofu_stall", "豆腐摊位，豆腐箱与石磨", "a tofu stall with tofu boxes and a stone mill", "石磨豆腐", "简体中文"),
 ("bg_conference", "国际会议中心走廊，玻璃幕墙", "a conference centre corridor with glass curtain walls", "会議室", "日语"),
 ("bg_boutique_floor", "服装店销售区，衣架与试衣间帘", "a boutique sales floor with racks and fitting room curtains", "新款上市", "简体中文"),
 ("bg_tailor_shop", "裁缝铺，缝纫机与布卷", "a tailor shop with sewing machines and fabric bolts", "裁縫店", "繁体中文"),
 ("bg_bakery", "烘焙店后厨，烤箱与操作台", "a bakery kitchen with ovens and work tables", "BAKERY", "英语"),
 ("bg_salon_front", "美容院前台，镜面与绿植", "a beauty salon reception with mirrors and plants", "Salon de Beauté", "法语"),
 ("bg_dessert_bar", "甜品店操作台，裱花与蛋糕展示柜", "a dessert counter with piping tools and a cake display", "甜品屋", "简体中文"),
 ("bg_training_centre", "家政培训中心，实操台与模型", "a domestic service training centre with practice stations", "职业培训", "简体中文"),
 ("bg_insurance", "保险公司办公室，档案柜与绿植", "an insurance office with filing cabinets and plants", "保险公司", "简体中文"),
 ("bg_small_restaurant", "街边小饭馆门口，塑料桌椅与蒸汽", "outside a small street restaurant with plastic tables and steam", "家常小館", "繁体中文"),
 ("bg_baking_class", "烘焙教室，长案与烤箱", "a baking classroom with long counters and ovens", "烘焙课程", "简体中文"),
 ("bg_support_centre", "电商客服中心，成排工位与电脑", "an e-commerce support centre with rows of desks and computers", "客服中心", "简体中文"),
 ("bg_flower_shop", "花店店里，满墙鲜花与花桶", "the inside of a flower shop with walls of blooms and buckets", "Fleur", "法语"),
 ("bg_senior_centre", "社区养老驿站，休息区与活动室", "a community senior centre lounge and activity room", "敬老院", "简体中文"),
 ("bg_boarding_gate", "机场登机口，晨光中的飞机舷窗", "an airport boarding gate with morning light through plane windows", "登机口", "简体中文"),
 ("bg_pet_shop", "宠物店，笼舍与宠物用品架", "a pet shop with cages and supply racks", "宠物店", "简体中文"),
 ("bg_grain_depot", "粮站仓库门口，堆叠的粮袋", "a grain depot entrance with stacked sacks", "粮食储备", "简体中文"),
 ("bg_driving_office", "驾校办公楼，墙上挂着英文标牌", "a driving school office with an English sign on the wall", "DRIVE SAFE", "英语"),
 ("bg_school_field", "学校操场，篮球架与跑道", "a school sports field with basketball hoops and a running track", "運動会", "繁体中文"),
 ("bg_auto_shop", "汽车修理厂，升起的车与工具墙", "an auto repair shop with a raised car and a tool wall", "整備工場", "日语"),
 ("bg_sorting_centre", "快递分拣中心，传送带与成堆包裹", "a parcel sorting centre with conveyors and heaps of packages", "分拣中心", "简体中文"),
 ("bg_cargo_deck", "远洋货轮甲板，远处海平线与集装箱", "the deck of an ocean cargo ship with a distant horizon and containers", "OCEAN STAR", "英语"),
 ("bg_cold_store", "肉联厂冷库门口，白霜与不锈钢推车", "the door of a meat processing cold store with frost and steel trolleys", "冷链仓库", "简体中文"),
 ("bg_tea_prep", "奶茶店后厨备料区，操作台与料桶", "a drink prep area with a work counter and ingredient bins", "調製室", "繁体中文"),
 ("bg_test_field", "试验田边，玉米植株与土壤取样盒", "the edge of a test field with corn plants and soil sampling boxes", "试验田", "简体中文"),
 ("bg_dock_market", "农贸市场入口，成排摊位与遮阳棚", "the entrance of a farmers market with rows of stalls and awnings", "農貿市場", "繁体中文"),
 ("bg_harbour_office", "港口调度室，大屏幕与海图", "a port dispatch office with large screens and nautical charts", "港口調度", "简体中文"),
 ("bg_apothecary", "中药铺，柜台与成排药抽屉", "a traditional Chinese apothecary with a counter and rows of medicine drawers", "同德堂藥號", "繁体中文"),
 ("bg_canteen", "工厂食堂，长条餐桌与打饭窗口", "a factory canteen with long dining tables and a serving window", "职工食堂", "简体中文"),
 ("bg_warehouse_rack", "大型仓库，货架与叉车", "a large warehouse with racking and forklifts", "立体仓库", "简体中文"),
 ("bg_bakery_shop", "面包店，玻璃柜台与面包架", "a bakery with a glass display counter and bread racks", "Boulangerie", "法语"),
 ("bg_gym_reception", "健身房前台，器械墙与电视", "a gym reception with a wall of equipment and a television", "健身會所", "繁体中文"),
 ("bg_car_repair_bay", "汽车美容店，抛光设备与洗车位", "an auto detailing shop with polishing equipment and a wash bay", "洗車場", "繁体中文"),
 ("bg_bookstore", "大型书店，层层书架与阅读区", "a large bookstore with tall shelves and a reading area", "读书斋", "简体中文"),
 ("bg_subway_station", "地铁站台，屏蔽门与广告灯箱", "a subway platform with platform screen doors and illuminated ad boxes", "地铁站台", "简体中文"),
 ("bg_night_market", "夜市，摊位灯串与人群光影", "a night market with strings of stall lights", "夜市", "繁体中文"),
 ("bg_construction_site", "建筑工地，塔吊与脚手架", "a construction site with tower cranes and scaffolding", "在建工程", "简体中文"),
 ("bg_harbour_fishing", "远洋渔港，冷藏仓与冰块", "a deep-sea fishing port with cold storage and ice blocks", "遠洋漁港", "繁体中文"),
 ("bg_restaurant_private", "私房菜包间，红木桌椅与字画", "a private dining room in a Chinese restaurant with rosewood furniture and calligraphy", "雅聚轩", "简体中文"),
 ("bg_tea_house", "中式茶馆，紫砂壶与竹帘", "a traditional Chinese tea house with clay teapots and bamboo blinds", "清心茶社", "简体中文"),
 ("bg_clinic_ward", "医院病房，病床与输液架", "a hospital ward with beds and IV stands", "住院部", "简体中文"),
 ("bg_school_gate", "中学校门口，牌坊与横幅", "the gate of a middle school with an archway and a banner", "实验中学", "简体中文"),
 ("bg_factory_dorm", "工厂宿舍楼，晾衣绳与自行车", "a factory dormitory block with washing lines and bicycles", "幸福公寓", "简体中文"),
 ("bg_market_alley", "老城巷弄，青砖墙与红灯笼", "an old city alley with grey brick walls and red lanterns", "福安巷", "简体中文"),
 ("bg_highspeed_station", "高铁站大厅，座椅与电子屏", "a high-speed rail station hall with seating and electronic boards", "候車大廳", "繁体中文"),
]


def blank_png(spec):
    import struct, zlib, tempfile
    w, h = map(int, spec.split(':', 1)[1].lower().split('x'))
    raw = b''.join(b'\x00' + b'\x80' * (3 * w) for _ in range(h))
    chunk = lambda t, d: struct.pack('>I', len(d)) + t + d + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    path = os.path.join(tempfile.gettempdir(), f'blank_{w}x{h}.png')
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
                + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))
    return path


def up(path):
    """上传文件到 ComfyUI。

    统一转成 RGB JPEG：ComfyUI 新版 LoadImage 走 pyav 的 VideoFromFile 路径，
    该路径按视频帧解码 PNG 会丢 alpha 通道，下游报
    "cannot reshape array of size N into shape (H,W,3)"。
    文件名单独唯一化，避免多实例并发上传同名文件时互相覆盖。
    """
    import uuid, shutil, tempfile
    from PIL import Image
    src = blank_png(path) if path.startswith('blank:') else path
    tmpdir = tempfile.mkdtemp(prefix='cimg_')
    try:
        dst = os.path.join(tmpdir, 'img.jpg')
        with Image.open(src) as im:
            im.convert('RGB').save(dst, quality=96, subsampling=0)
        return comfy.upload(dst)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


UNET = 'qwen_image_edit_2511_int8_convrot.safetensors'
CLIP = 'qwen_2.5_vl_7b_fp8_scaled.safetensors'
VAE = 'qwen_image_vae.safetensors'
LORA = 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors'


def build_graph(img, prompt, seed, prefix, neg):
    g = {
        '41': {'class_type': 'LoadImage', 'inputs': {'image': img}},
        '161': {'class_type': 'UNETLoader', 'inputs': {
            'unet_name': comfy.find_model('diffusion_models', UNET), 'weight_dtype': 'default'}},
        '162': {'class_type': 'CLIPLoader', 'inputs': {
            'clip_name': comfy.find_model('text_encoders', CLIP), 'type': 'qwen_image', 'device': 'default'}},
        '146': {'class_type': 'VAELoader', 'inputs': {'vae_name': comfy.find_model('vae', VAE)}},
        '160': {'class_type': 'FluxKontextImageScale', 'inputs': {'image': ['41', 0]}},
        '151': {'class_type': 'TextEncodeQwenImageEditPlus', 'inputs': {
            'clip': ['162', 0], 'vae': ['146', 0], 'image1': ['160', 0], 'prompt': prompt}},
        '149': {'class_type': 'TextEncodeQwenImageEditPlus', 'inputs': {
            'clip': ['162', 0], 'vae': ['146', 0], 'image1': ['160', 0], 'prompt': neg}},
        '148': {'class_type': 'FluxKontextMultiReferenceLatentMethod', 'inputs': {
            'conditioning': ['151', 0], 'reference_latents_method': 'index_timestep_zero'}},
        '147': {'class_type': 'FluxKontextMultiReferenceLatentMethod', 'inputs': {
            'conditioning': ['149', 0], 'reference_latents_method': 'index_timestep_zero'}},
        '145': {'class_type': 'ModelSamplingAuraFlow', 'inputs': {'model': ['161', 0], 'shift': 3.1}},
        '152': {'class_type': 'CFGNorm', 'inputs': {'model': ['145', 0], 'strength': 1.0}},
        '153': {'class_type': 'LoraLoaderModelOnly', 'inputs': {
            'model': ['152', 0], 'lora_name': comfy.find_model('loras', LORA), 'strength_model': 1.0}},
        '156': {'class_type': 'VAEEncode', 'inputs': {'pixels': ['160', 0], 'vae': ['146', 0]}},
        '169': {'class_type': 'KSampler', 'inputs': {
            'model': ['153', 0], 'positive': ['148', 0], 'negative': ['147', 0],
            'latent_image': ['156', 0], 'seed': seed, 'steps': 4, 'cfg': 1.0,
            'sampler_name': 'euler', 'scheduler': 'simple', 'denoise': 1.0}},
        '158': {'class_type': 'VAEDecode', 'inputs': {'samples': ['169', 0], 'vae': ['146', 0]}},
        '195': {'class_type': 'SaveImage', 'inputs': {'images': ['158', 0], 'filename_prefix': prefix}},
    }
    return g


def run_one(img, prompt, seed, prefix, out_dir, neg):
    g = build_graph(img, prompt, seed, prefix, neg)
    e = comfy.run(g)
    err = comfy.error(e)
    if err:
        raise RuntimeError(err)
    files = []
    for o in e.get('outputs', {}).values():
        for it in o.get('images', []):
            if it.get('type', 'output') != 'output':
                continue
            q = urllib.parse.urlencode({'filename': it['filename'], 'subfolder': it.get('subfolder', ''),
                                        'type': 'output'})
            base = it['filename']
            if it.get('subfolder'):
                base = os.path.join(it['subfolder'], it['filename'])
            dst = os.path.join(out_dir, base)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with urllib.request.urlopen(f"{comfy.URL}/view?{q}", timeout=600) as r, open(dst, 'wb') as f:
                f.write(r.read())
            files.append(dst)
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--of', type=int, default=1)
    ap.add_argument('--url', default=None)
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()
    if args.url:
        os.environ['COMFY_URL'] = args.url
        import importlib
        importlib.reload(comfy)

    os.makedirs(args.out, exist_ok=True)
    log = open(f"{args.out}/shard{args.shard}.log", 'a', encoding='utf-8')

    def say(*a):
        msg = ' '.join(str(x) for x in a)
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    mine = [(i, s) for i, s in enumerate(SCENES) if i % args.of == args.shard]
    say(f'=== shard {args.shard}/{args.of}: {len(mine)} scenes on {comfy.URL} ===')

    for i, (sid, cn, en, text, lang) in mine:
        dst = f"{args.out}/{sid}.png"
        if os.path.exists(dst) and os.path.getsize(dst) > 20000:
            say(f'  {sid} exists, skip')
            continue
        prompt = (f'A photorealistic cinematic background of {en}. '
                  f'The scene is empty of people. '
                  f'There is a prominent clear sign in the scene displaying exactly this text: "{text}". '
                  f'The text is written in {lang}, every character is correct and legible, '
                  f'rendered with sharp clean strokes on a realistic signboard, '
                  f'spelled correctly, no misspelled letters, no random symbols. '
                  f'{STYLE}')
        seed = 770000 + i * 37
        ok = False
        for attempt in range(3):
            try:
                fs = run_one(up('blank:1216x832'), prompt, seed + attempt, f'scene_{sid}', args.out, NEG)
                if fs:
                    os.replace(fs[0], dst)
                    ok = True
                    say(f'  {sid} -> {dst} [{lang}: {text}]')
                    break
            except Exception as ex:
                say(f'  {sid} attempt {attempt} failed: {ex}')
                time.sleep(3)
        if not ok:
            say(f'  {sid} FAILED')

    say('=== shard finished ===')


if __name__ == '__main__':
    main()
