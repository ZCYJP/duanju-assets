#!/bin/bash
# 批量 OCR 参考片（远端抽的 2fps 帧）+ 我们成片帧，输出带位置/字号
set -u
cd /tmp/refan
echo "### REF ###" > /tmp/refan/ocr_ref.txt
ls /tmp/refan/frames/*.jpg | head -1600 | xargs -P 8 -n 40 /tmp/refan/mac_ocr >> /tmp/refan/ocr_ref.txt 2>/dev/null
echo "### OURS ###" > /tmp/refan/ocr_ours.txt
ls /ssd2/duanju/drama_1min/qc_final_v2/*.png | xargs -P 8 -n 40 /tmp/refan/mac_ocr >> /tmp/refan/ocr_ours.txt 2>/dev/null
echo "DONE"
wc -l /tmp/refan/ocr_ref.txt /tmp/refan/ocr_ours.txt
