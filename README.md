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
| `characters/cast/` | 角色主形象与设定图 | **150** 组 | 1.50 G |
| `characters/sheets/` | 角色设定对照表 | 1 组 | 0.50 G |
| `scenes_100/` | 剧情背景（含真实汉字招牌） | **100** 张 PNG | 0.10 G |
| `drama/` | 短剧《深渊》分镜与成片片段 | 35 个文件 | 0.06 G |
| `scripts/` | 全部生成脚本（可复现） | 153 个文件 | 1.2 M |
| `*.html` | 三个可视化索引页 | — | 1.9 M |

**去重后净入库体积：约 6.8 G / 7128 个文件。**

---

## 生成技术栈

| 环节 | 模型 / 工具 |
|---|---|
| 形象与背景生成 | **Qwen-Image-Edit-2511**（int8 convrot）+ **Lightning 4 步 LoRA**，cfg=1.0 |
| 形象一致性 | 以角色主形象为参考图做 image-edit，同一张脸派生 26 个变体 |
| 音色生成 | **Qwen3-TTS-12Hz-1.7B-VoiceDesign**（文本描述式音色设计） |
| 情绪音高校准 | librosa 相位声码器变调 + **相对锚定**判据（见下） |
| 文字核验 | easyocr 1.7.2 (GPU) + tesseract 4.1.1 + **Qwen2.5-VL-7B** 四层裁决 |

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

> 初版规则把**空格 U+0020 也算作非法字符**，导致所有多词英文 / 法语招牌
> （如 "DRIVE SAFE"、"Salon de Beauté"）被系统性误判为伪文字；
> 另一处缺陷是 VL 返回候选标签列表 `[REAL]/[FAKE]/[CJK]` 时被当作"同时命中"，
> 使实际完全正确的招牌被判伪。两者均已修正并重跑全量。

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

---

## 许可

代码与数据均采用 **MIT** 许可，详见 `LICENSE`。
模型权重版权归原作者所有，本仓库不包含任何模型权重。
