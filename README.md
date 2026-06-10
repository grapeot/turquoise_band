# turquoise_band — 月全食绿松石带的物理重现

月全食时，月面从本影深处的暗红、过渡到一条青绿色窄带（**绿松石带 / turquoise band**），再到边缘的正常月光白。本项目从辐射传输第一性出发，定量重现这条带，最终渲染合成月食照片。

**站点（图、视频与六步推演）**：https://grapeot.github.io/turquoise_band/ ｜ 原理深度解读见站点 principles 页。

物理直觉：折射进地球本影的阳光掠过大气 limb，沿月面径向对应不同擦边高度。低高度处瑞利散射（∝λ⁻⁴）滤掉蓝光留红 → 血月；平流层高度处臭氧 Chappuis 带（~500–700nm）吃掉橙红 → 绿松石；高高度处光程薄 → 接近原始日光白。

## 状态

- [x] **L0 辐射传输闭环**：色相曲线红→青→白，实测数据（O₃ Serdyuchenko 2014、AFGL 大气、SAO2010 太阳谱、CIE CMF）
- [x] **L1 折射几何（解析）**：单一映射 `r(h)=(R⊕+h)−α·d_moon`，对照 Mallama Table 3.1 中高 h 吻合<2%（低 h 端 ~6-10%）
- [x] **L2 逐像素反向 RT 渲染**（render_rt.py）：分支感知反查，无 banding，numpy 亚秒级
- [x] **L3 写实月盘**：NASA 月面纹理 × 物理食光颜色，对数 tone map，忠实亮度（蓝带物理真实地窄）
- [x] **太阳圆盘 ray tracing**（brute_ray_trace.py）：32' 太阳圆盘比带宽 5 倍，把点源版过饱和的浓青(0.41)柔化到符合观测的浅青(0.70)
- [x] **真·正向 ray tracing**（raytrace_eclipse.py，当前权威管线）：真折射 RK4 + 弯曲路径消光（用真实切点高度）+ 撒线 focusing 涌现 + 半影直射光，零解析处方。本影中心 ≈−13.5 档
- [x] **HDR gain map**：可下 iPhone Photos 的 HEIC（tools/make_gainmap_hdr.swift）

### 核心物理结论
绿松石带本质是**青色（cyan/teal）**、**淡**、**窄**：颜色（R/B）沿月盘是平滑渐变，但
**"蓝带窄"是亮度现象**——最蓝处亮度比满月暗 ~10 档，被旁边暴亮的趋白区盖过，
真实可见只剩一道细带。当前模型最蓝 R/B≈0.70@41'（raw sRGB 口径）；按 Shu 2024 的
654/491 卫星口径折算后 in-band 0.86-1.0，落在其实测 0.8-1.0 区间（注意两种口径不可直接混比，
详见 working.md 续16 与 2026-06-09 勘误）。网上"半盘饱和蓝"是后期产物。
本影中心暗端与 Mallama clear-sky 模型的差距已对账定位（消光标定口径差，见 working.md 2026-06-09/06-10）。

### 科学边界

食光颜色与亮度由辐射传输决定，屏幕观感由曝光与 tone map 决定，两套体系严格分开——本项目曾两次差点因显示层误判物理（详见 working.md Lessons）。模型的适用范围：

- 本影中心 −15.1 档是**最晴夜背景气溶胶口径**（分子大气理论上限 −13.5 档）；真实深食随当时平流层气溶胶可以暗得多。**本模型不能告诉你下一次月食是亮是暗——那由当时的平流层气溶胶决定。**
- Shu 2024 卫星窄带口径的 R/B<1 ribbon 在一维径向平均下复现不出（红肩消光/口径差，开放项，见 docs/MODEL_CARD.md）。
- I 波段缺水汽/氧气吸收带，绝对值不引用。

### 主要输出（outputs/）
- `moon_disk_turquoise_final.png` — 写实月食照片（圆盘物理，d=40 最蓝处过月盘中心）
- `moon_composite.png` — 剖面诊断（量化蓝带宽度）
- `moon_brightness_cliff.png` — 两种曝光对比（揭示"蓝带窄是亮度现象"）
- `ablation/step_*.tif` — 六步机制 ablation（土圆盘→遮挡→瑞利→臭氧→太阳圆盘→真 ray tracing）
- `moon_eclipse_sdr/hdr_h265.mp4` — 月食对偶视角视频（月面 | 地球全景 | 大气环特写 | 光谱成因）

### 跑
```bash
source .venv/bin/activate
python src/raytrace_eclipse.py         # 真·正向 ray tracing（权威管线，~3s）
python -m pytest src/tests/ -v         # 物理不变量 + 渲染冒烟测试（含 slow，~2min）
python src/render_textured.py          # 写实月盘（默认 d=40、权威 ray tracing LUT；--engine pointsource 仅历史对照）
python src/diag_composite.py           # 蓝带宽度诊断
python src/pipeline.py --self-check    # L0 色相曲线
bash scripts/make_hdr.sh               # 出 HDR HEIC（需 HDR_REFERENCE_HEIC 指向任一 iPhone 实拍 HEIC）
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

机器：Apple M3 Ultra（32 CPU / 80 GPU / 512GB）。全管线 numpy 矢量化 CPU（完整 ray tracing ~3s），未用 MPS。
