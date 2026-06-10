#!/bin/bash
# 一条命令出带 gain map 的 HDR HEIC（下 iPhone Photos 可见 HDR）。
# 流程：Python 渲染 SDR base + 线性 HDR TIFF → Swift CLI 算 gain map 写 HEIC。
set -e
cd "$(dirname "$0")/.."

# gain map 元数据模板：任何 iPhone 默认相机直出的 HEIC 都可以（只取其 HDR gain map
# 元数据结构，不用其图像内容）。用环境变量指定：
#   HDR_REFERENCE_HEIC=~/Pictures/IMG_0001.HEIC bash scripts/make_hdr.sh
REF="${HDR_REFERENCE_HEIC:-../YabiVision/YabiVision/Resources/reference.HEIC}"
[ -f "$REF" ] || { echo "缺 gain map 元数据模板。请设 HDR_REFERENCE_HEIC=<任一 iPhone 实拍 HEIC>（需含 Apple gain map，iPhone 默认相机直出即可）"; exit 1; }

echo "[1/2] 渲染 SDR base + 线性 HDR..."
source .venv/bin/activate
python src/render_textured.py "$@"

echo "[2/2] 生成 gain map HEIC..."
swift tools/make_gainmap_hdr.swift \
  outputs/moon_realistic_raw.png \
  outputs/moon_hdr_linear.tif \
  "$REF" \
  outputs/moon_hdr.heic

echo ""
echo "完成：outputs/moon_hdr.heic"
echo "AirDrop 到 iPhone，在 Photos 打开看亮部超亮（HDR 仅在物理 iPhone+HDR屏可见）。"
