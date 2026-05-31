# turquoise_band — 月全食绿松石带的物理重现

月全食时，月面从本影深处的暗红、过渡到一条青绿色窄带（**绿松石带 / turquoise band**），再到边缘的正常月光白。本项目从辐射传输第一性出发，定量重现这条带，最终渲染合成月食照片。

物理直觉：折射进地球本影的阳光掠过大气 limb，沿月面径向对应不同擦边高度。低高度处瑞利散射（∝λ⁻⁴）滤掉蓝光留红 → 血月；平流层高度处臭氧 Chappuis 带（~500–700nm）吃掉橙红 → 绿松石；高高度处光程薄 → 接近原始日光白。

## 状态

- [x] **L0 辐射传输闭环**：色相曲线红→青→白，实测数据（O₃ Serdyuchenko 2014、AFGL 大气、SAO2010 太阳谱、CIE CMF）
- [x] **L1 折射几何**：单一映射 `r(h)=(R⊕+h)−α·d_moon`，对照 Mallama Table 3.1 吻合<2%
- [x] **L2 逐像素反向 RT 渲染**（render_rt.py）：分支感知反查，无 banding，numpy 亚秒级
- [x] **L3 写实月盘**：NASA 月面纹理 × 物理食光颜色，对数 tone map，忠实亮度（蓝带物理真实地窄）
- [x] **HDR gain map**：可下 iPhone Photos 的 HEIC（tools/make_gainmap_hdr.swift）

### 核心物理结论（经卫星实测 + 文献验证）
绿松石带本质是**青色（cyan/teal）**、**淡**、**窄**：颜色（R/B）沿月盘是平滑渐变，但
**"蓝带窄"是亮度现象**——最蓝处亮度仅月盘最亮的 ~4.5%，被旁边暴亮的趋白区盖过，
真实可见蓝带 ~1-2 arcmin（与 Shu 2024 卫星实测 R/B 0.8-1.0 一致）。网上"半盘饱和蓝"是后期产物。

### 主要输出（outputs/）
- `moon_realistic_raw.png` — 写实月食照片（忠实亮度，d=47 蓝带过月盘中心）
- `moon_composite.png` — 剖面诊断（量化蓝带宽度）
- `moon_brightness_cliff.png` — 两种曝光对比（揭示"蓝带窄是亮度现象"）

### 跑
```bash
source .venv/bin/activate
python src/render_textured.py          # 写实月盘
python src/diag_composite.py           # 蓝带宽度诊断
python src/pipeline.py --self-check    # L0 色相曲线
bash scripts/make_hdr.sh               # 出 HDR HEIC
```

## 文档

- `docs/working.md` — **当前权威记录**：Changelog + Key Technical Decisions + Lessons（推荐先读）
- `docs/PHYSICS_AND_PITFALLS.md` — 物理方法 + 踩坑全记录
- `docs/L1_geometry.md` — 折射几何处方（单一映射，对照 Mallama）
- `docs/PRD.md` / `docs/RFC.md` — 目标 / 技术设计
- `docs/SCIENCE_REVIEW.md` — 文献调研摘要
- `docs/LOG.md` — 早期日志（归档，部分结论已被 working.md 取代）

## 环境

```bash
source .venv/bin/activate   # uv venv, Python 3.12
```

机器：Apple M3 Ultra（32 CPU / 80 GPU / 512GB），渲染阶段走 MPS。
