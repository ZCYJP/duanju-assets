#!/usr/bin/env bash
# ============================================================
# 短剧数字人资产库 —— GitHub 上架脚本（远程 h100-038 上执行）
#
# 用法：
#   bash tmp_gh_commit.sh              # 只做 add + commit（不推送）
#   bash tmp_gh_commit.sh --push       # add + commit + push
#
# 前置：
#   - 已在 /ssd2/duanju 执行 git init / git lfs install --local
#   - .gitignore .gitattributes README.md LICENSE 已就位
#   - 已配置 remote（见脚本末尾提示）
# ============================================================
set -euo pipefail

cd /ssd2/duanju

PUSH=0
if [[ "${1:-}" == "--push" ]]; then PUSH=1; fi

echo "=== [1/6] 检查 Git LFS ==="
git lfs version
git lfs track "*.png" "*.wav" "*.mp4" "*.jpg" "*.jpeg" "*.mkv" 2>/dev/null || true

echo
echo "=== [2/6] 暂存 ==="
git add -A

echo
echo "=== [3/6] 统计 ==="
STAGED=$(git diff --cached --name-only | wc -l)
echo "暂存文件数: $STAGED"
echo "按类型:"
git diff --cached --name-only | grep -oE '\.[A-Za-z0-9]+$' | sort | uniq -c | sort -rn | head -12 || true

echo
echo "LFS 追踪文件数:"
git lfs ls-files | wc -l

echo
echo "=== [4/6] 体积核对 ==="
echo "仓库 .git 目录当前占用:"
du -sh .git 2>/dev/null || true

echo
echo "=== [5/6] 提交 ==="
git commit -q -m "短剧数字人资产库：150 角色 / 3900 时令形象 / 1350 情绪音色 / 100 背景

形象：Qwen-Image-Edit-2511 + Lightning 4步，150 角色 × 26 变体（冬夏+二十四节气）
音色：Qwen3-TTS VoiceDesign，150 角色 × 9 情绪，相对锚定音高校准 98.30% 达标
背景：100 张，招牌文字经 OCR + VL 四层核验，全部为真实汉字/外文
成片：短剧《深渊》分镜与 1 分钟样片

二进制文件走 Git LFS；ComfyUI 中间产物、备份与归档副本已由 .gitignore 排除。

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" \
  && echo "提交完成" || echo "无变更可提交"

git log --oneline -1 2>/dev/null || true

echo
echo "=== [6/6] 推送 ==="
if [[ $PUSH -eq 1 ]]; then
    if git remote -v | grep -q .; then
        git push -u origin HEAD
        echo "推送完成"
    else
        echo "!! 未配置 remote，先执行："
        echo "   git remote add origin https://github.com/<owner>/<repo>.git"
        echo "   git lfs env > /tmp/lfs-env.txt   # 若需配置 LFS 认证"
        echo "   bash tmp_gh_commit.sh --push"
    fi
else
    echo "跳过推送（加 --push 参数可推送）"
    git remote -v || echo "（未配置 remote）"
fi
