# turquoise_band — 月全食绿松石带的物理重现

月全食时，月面从本影深处的暗红、过渡到一条青绿色窄带（**绿松石带 / turquoise band**），再到边缘的正常月光白。本项目从辐射传输第一性出发，定量重现这条带，最终渲染合成月食照片。

物理直觉：折射进地球本影的阳光掠过大气 limb，沿月面径向对应不同擦边高度。低高度处瑞利散射（∝λ⁻⁴）滤掉蓝光留红 → 血月；平流层高度处臭氧 Chappuis 带（~500–700nm）吃掉橙红 → 绿松石；高高度处光程薄 → 接近原始日光白。

## 状态

- [x] Scaffold（repo / venv / PRD / RFC）
- [x] Science review（多 agent 调研，见 `docs/SCIENCE_REVIEW.md`）
- [x] 下载权威数据（O₃ Serdyuchenko 2014 截面、AFGL 大气廓线、CIE CMF）
- [x] **L0：色相曲线闭环 ✓** —— 红→青→白趋势 + 亮度暴跌 240× 全部自查通过（`outputs/L0_colorband.png`）
- [ ] L1：定量校核（真实折射几何 + air-mass 增强臭氧柱，把青绿色相从偏蓝拉正）
- [ ] L2：合成月食照片渲染（月盘几何 + MPS）

L0 结果：合成色带肉眼可见 **深暗红（本影深处，2-14km）→ 青绿 teal 带（16-36km）→ 白边（36km+）**。
跑：`source .venv/bin/activate && python src/pipeline.py --self-check`

## 文档

- `docs/PHYSICS_AND_PITFALLS.md` — **物理方法 + 踩坑全记录**（推荐先读）
- `docs/PRD.md` — 现象、目标、成功标准、数据需求
- `docs/RFC.md` — 计算管线技术设计、模块划分、验证钩子
- `docs/L1_geometry.md` — 折射几何处方（含红核/绿松石带双 limb 修正）
- `docs/SCIENCE_REVIEW.md` — 文献调研摘要
- `docs/LOG.md` — 时间线日志

## 环境

```bash
source .venv/bin/activate   # uv venv, Python 3.12
```

机器：Apple M3 Ultra（32 CPU / 80 GPU / 512GB），渲染阶段走 MPS。
