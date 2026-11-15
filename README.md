# 短剧数字人资产库

面向中文短剧生产的 AI 生成资产：**150 个角色形象**、**3900 张时令形象变体**、
**1350 条情绪音色**、**100 张剧情背景**。全部在单机 8 卡节点上生成。

> ⚠️ 本仓库的图像、音频、角色设定与剧本均为 **AI 生成的虚构内容**，
> 人物形象不对应任何真实人物。请勿用于伪造、冒充或误导。

---

## 资产总览

| 目录 | 内容 | 数量 | 体积 |
|---|---|---|---|
| `seasons/` | 每角色在冬 / 夏 + 二十四节气共 26 个场景下的形象变体 | 150 × 26 = **3900** 张 PNG | 4.08 G |
| `voices/emotions/` | 每角色 9 种情绪音色（喜/怒/哀/乐/惧/惊/厌/忧/羞） | 150 × 9 = **1350** 条 WAV | 0.46 G |
| `characters/cast/` | 角色主形象与设定图（每角色 10 张：hero / headshot / fullbody / back / 三视图等） | **150** 组 = 1500 张 PNG | 1.50 G |
| `characters/sheets/` | 角色设定对照表 | 1 组 | 0.50 G |
| `scenes_100/` | 剧情背景（招牌文字 100/100 全部为真实汉字） | **100** 张 PNG | 0.10 G |
| `drama/` | 短剧《深渊》分镜与成片片段 | 35 个文件 | 0.06 G |
| `scripts/` | 全部生成脚本（可复现） | 153 个文件 | 1.2 M |
| `*.html` | 三个可视化索引页 | — | 1.9 M |

**净入库：7482 个 LFS 对象 + 236 个文本文件 = 7718 个文件 / 约 6.8 G。**

> 磁盘上的 `seasons/`、`characters/cast/` 另存有 `_backup_gender/` 等回滚备份，
> 已由 `.gitignore` 排除，**不入库**。入库数量以 `git ls-files` 为准。

---

## 生成技术栈

| 环节 | 模型 / 工具 |
|---|---|
| 形象与背景生成 | **Qwen-Image-Edit-2511**（int8 convrot）+ **Lightning 4 步 LoRA**，cfg=1.0 |
| 形象一致性 | 以角色主形象为参考图做 image-edit，同一张脸派生 26 个变体 |
| 形象性别自检 | **Qwen2.5-VL-7B** 无偏性别判定 + 三视图多数投票（见下） |
| 音色生成 | **Qwen3-TTS-12Hz-1.7B-VoiceDesign**（文本描述式音色设计） |
| 情绪音高校准 | librosa 相位声码器变调 + **相对锚定**判据（见下） |
| 文字核验 | easyocr 1.7.2 (GPU) + tesseract 4.1.1 + **Qwen2.5-VL-7B** 四层裁决 |

### 角色形象的性别一致性

出图 prompt 形如
`A photorealistic cinematic master portrait of {desc}, {scene}, {STYLE}`，
而角色 `desc` 中 **136 / 150（90.7%）不含 `woman` / `male` 等性别词**，
模型只能靠职业与体态自行猜测。方颌、短发、西装这类特征必然被渲染成男性，
导致 **12 / 150（8.0%）的形象与设定性别不符**。

修复分四层：

| 层 | 手段 | 说明 |
|---|---|---|
| L1 | 补 `desc` 性别词 | 98 个角色在职业词前注入 `female` / `male`，复检 150/150 无缺失 |
| L2 | GENDER LOCK 提示词加固 | positive 前置「This is unmistakably a woman/man…」；negative 排除相反性别特征（方颌 / 胡茬 / 宽肩 / 妆容…） |
| L3 | 生成 → 自检 → 择优闭环 | 每角色按 `base_seed + k×1000` 换种子生成 4–5 个候选，VL 判定后择优；派生视图同样过筛，不合格换种子重试 |
| L4 | ID 语义纠正 | 15 个女性角色原用 M 段编号（M37–M68），重命名为 F71–F85，同步角色目录、26 张变体、9 条音色与清单 |

**自检方法（关键）**：用 Qwen2.5-VL-7B 做**无偏性别判定**——
提示词只问「画面中的人看起来是男性还是女性」，**不透露设定性别**，
避免模型顺着提示词附和。取 hero / headshot / fullbody 三视图多数投票，
平票取 hero。背面图（back）**不参与判定**：看不到脸，
VL 只能靠「短发 + 工装」猜，会系统性偏向男性。

**验收结果：150 / 150 形象与设定性别一致，不符 0 例。**

> 附带修复：`dh_scripts/comfy.py` 的 `run()` 原本 `except Exception: continue`，
> ComfyUI 实例重启后 `/history/{pid}` 永不出现，单次 60s 超时被吞掉形成**无限死等**
> （实测挂 20+ 分钟，队列全空、GPU 0%）。已加总超时 900s、连续异常上限 8、
> 异常时探测 `/system_stats` 判存活。该修复对所有出图任务生效。

### ID 命名规范

角色 ID 前缀必须与设定性别一致：`F` = 女（Female），`M` = 男（Male）。
`cast_all_150.json` 的 `gender` 字段为唯一权威来源，
`characters/cast/`、`seasons/`、`voices/emotions/` 三处目录名均由 ID 派生。
本次已将 15 个误用 M 段编号的女性角色统一改为 F71–F85（详见 `characters/rename_log.json`）。

### 情绪音色的音高校准方法

用户的 9 情绪声音特征表给的是**倍率区间**（相对角色基线音），不是绝对频率。
直接匹配绝对 f0 会陷入八度判定的死局（实测 pyin 在超低音区会锁到高次谐波，
M02_yan 真值约 90 Hz 却读成 920.9 Hz）。

最终方案是**相对锚定**：以角色基线音为锚，计算
`r = f0(情绪) / f0(基线)`，比值对八度错误免疫；再用相位声码器把 `r` 修正到目标区间。
f0 估计采用 pyin（经 240 对真实人声 × 4 个变调因子的往返测试，
八度正确率 95.8%、误差中位 1.46%，为四种候选估计器中最优）。

验收结果：**1327 / 1350 = 98.30%** 落在目标倍率区间；
独立三链复验 97.8%；基线自校准一致率 100%。

### 背景文字真实性核验

背景里的招牌文字必须是真的汉字（简 / 繁 / 日 / 韩 / 英 / 法均可），不能用 AI 乱码充数。
采用四层证据链：

| 层 | 方法 | 产出 |
|---|---|---|
| L1 | easyocr + tesseract 双引擎，形态学定位文字区域并裁剪放大二次 OCR | `scenes_100_text_audit_v5.json` |
| L2 | VL 整图转写 | `scenes_100_text_vl_verify_v5.json` |
| L3 | VL 逐招牌标注「真实字符 / 伪文字」 | `scenes_100_text_signcheck_v5.json` |
| L4 | VL 4× 放大逐字符 + 拉丁串语义判定（是否为真实单词） | `scenes_100_text_zoomcheck_v5.json` / `_semantic_v5.json` |
| 裁决 | 四层融合 | `scenes_100_text_final_v5.json` |

字符集规则（v5）见 `scripts/tmp_text_rule_v5.py`，允许
空白字符、ASCII 字母数字、拉丁扩展字母（é/è/ç）、CJK / 假名 / 谚文；
`-` `|` `—` `/` `[` `]` 等装饰符号仍判为伪文字。

规则修复后共重生成 **11 张**背景（9 + bg_fresh_aisle + bg_boutique），
每张都经过四层核验复核才替换入库。

**最终结论：100 / 100 全部为真实文字，伪文字 0，无文字 0。**

### 修复过程中发现并修正的三处审计缺陷

| # | 缺陷 | 影响 | 修正 |
|---|---|---|---|
| 1 | 字符集规则把**空格 U+0020** 判为非法字符 | 所有多词英文/法语招牌（"DRIVE SAFE"、"Salon de Beauté"）被系统性误判为伪文字 | 允许空白字符 + 拉丁扩展字母（é/è/ç） |
| 2 | VL 返回**候选标签列表** `[REAL]/[FAKE]/[CJK]/[无法辨认]` 时被当作"同时命中" | 实际完全正确的招牌（如"石磨豆腐"）被判伪 | 识别候选列表格式，不提取任何标签 |
| 3 | `tmp_vl_verify.py --all` 用 `os.listdir(ROOT)` 扫目录 | 把 `text_report_*.json` 等非图片文件当成 sid 传入而崩溃 | 改用显式 sid 列表 |

> 缺陷 1 是系统性误伤：它在 v5 规则前把所有多词招牌都排除在"真实文字"之外，
> 正是这个假阴性把 `bg_driving_office`、`bg_boutique` 等误判成需要重生成。

---

## 索引页

| 文件 | 内容 |
|---|---|
| `cast_index.html` | 150 个角色的形象与设定对照 |
| `season_index.html` | 3900 张时令形象 + 1350 条情绪音色（可点图试听） |
| `scenes_index.html` | 100 张背景及逐张文字核验结论 |

直接在浏览器打开即可，无需服务器。

---

## 仓库体积与 Git LFS

净内容 6.8 G 超过 GitHub 单仓库 5 GB 建议上限，因此**图片与音频全部走 Git LFS**：

```bash
git lfs install
git lfs track "*.png" "*.wav" "*.mp4"
git add .gitattributes
```

`.gitignore` 已排除以下内容（不重复入库）：

- `comfy_out_*/` — ComfyUI 工作区中间产物（5.7 G），与 `seasons/` 内容重复
- `voices/emotions_backup_*/`、`voices/emotions_fixed*/` — 音色迭代版本与备份（1.4 G）
- `*.tar`、`*.tar.gz` — 归档副本（0.9 G），与目录内容重复
- `scenes_100_regen*/`、`scenes_100_pre_v5/` — 背景重生成工作区
- `pool/`、`pool_season/`、`logs/` — 任务池与运行日志
- `characters/cast/*/_backup_gender/`、`seasons/*/_backup_gender/` — 性别修复的回滚备份
- `drama_1min/sample_15s/{frames,motion,ocrframes,out,subs,tmp,voice}/` — 进行中任务的工作区

---

## 复现

生成脚本集中在 `scripts/`，按依赖顺序：

```
gen_cast.py            角色主形象 + 设定
gen_scenes_100.py      100 张背景
gen_season_variants.py 26 个时令变体（多卡并行）
gen_emotion_voices_v2.py  9 情绪音色
tmp_emotion_pitch_fix_v16.py  情绪音高相对锚定校准
```

审计与索引：

```
tmp_bg_text_audit.py         L1 OCR 审计
tmp_vl_verify.py             L2 VL 整图
tmp_vl_signcheck.py          L3 VL 招牌标注
tmp_vl_zoom.py               L4 VL 放大
tmp_vl_semantic.py           L4 拉丁语义
tmp_bg_text_adjudicate.py    四层裁决
tmp_text_rule_v5.py          字符集规则（v5 修正版）
build_cast_index.py          角色索引页
build_season_index.py        时令 + 情绪索引页
build_scenes_index.py        背景索引页
```

性别一致性：

```
tmp_fix_desc_gender.py       为 cast_all_150.json 的 desc 补性别词
tmp_patch_gen_cast_gender.py 给 gen_cast.py / gen_season_variants.py 打 GENDER LOCK
tmp_gender_audit.py          VL 无偏性别判定（全量 / --ids 定向）
tmp_regen_gender_mismatch.py 不符角色的「生成 → 自检 → 择优」闭环重生成
tmp_regen_seasons_for_ids.py 指定角色的 26 张时令变体重生成
tmp_rename_female_ids.py     15 个女角色 ID 由 M 段改为 F 段
```

---

## 许可

代码与数据均采用 **MIT** 许可，详见 `LICENSE`。
模型权重版权归原作者所有，本仓库不包含任何模型权重。
