#!/bin/bash
# 一条命令出带 gain map 的 HDR HEIC（下 iPhone Photos 可见 HDR）。
# 流程：Python 渲染 SDR base + 线性 HDR TIFF → Swift CLI 算 gain map 写 HEIC。
set -e
cd "$(dirname "$0")/.."

REF="../YabiVision/YabiVision/Resources/reference.HEIC"
[ -f "$REF" ] || { echo "缺 reference.HEIC（YabiVision 的 gain map 元数据来源）"; exit 1; }

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
