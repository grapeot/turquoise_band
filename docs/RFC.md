# RFC：绿松石带计算管线的技术设计

> **历史文档**：本文写于项目 scaffold 期（2026-05-31），记录当时的目标与设计。项目实际终点远超此范围（真·正向 ray tracing、对偶视角视频、ablation、HDR）。当前权威状态见 `working.md`（Key Technical Decisions）与 `MODEL_CARD.md`；对外结论层是 `README.md`。

状态：草案 v0。这份文档定义 L0（色相曲线闭环）的计算管线，并为 L2（渲染）预留接口。

## 总体管线

```
擦边高度 h (tangent height)
   │  几何：沿视线穿过大气的路径，每段对应一个 (高度, 路程) 采样
   ▼
柱密度积分  N_air(h, λ-indep) , N_O3(h)        ← 大气廓线 (数密度 vs 高度)
   │
   ▼
光学厚度  τ(λ, h) = ∫ [ n_air·σ_rayleigh(λ) + n_O3·σ_chappuis(λ) ] ds
   │      σ_rayleigh ∝ λ^-4 (含折射率色散修正)
   │      σ_chappuis(λ): 臭氧吸收截面，实测查表
   ▼
透射率谱  T(λ, h) = exp(-τ(λ, h))
   │
   ▼
出射谱  I(λ, h) = I_sun(λ) · T(λ, h)            I_sun: 太阳谱或黑体
   │
   ▼
颜色   XYZ = ∫ I(λ,h)·CMF(λ) dλ  →  sRGB, 色相角 hue, 亮度 Y
   │
   ▼
沿月面径向 r → 擦边高度 h(r)（本影几何映射）
   │
   ▼
L0 输出：hue(r), Y(r), sRGB(r) 曲线  ←★ 第一个闭环
L2 输出：月盘每像素 → h → sRGB，渲染成图
```

## 模块划分（src/）

- `atmosphere.py` — 大气模型：高度→空气数密度、臭氧数密度。先用 US Std Atm 1976 + 标准臭氧廓线；接口 `n_air(h)`, `n_o3(h)`。
- `cross_sections.py` — 截面：`sigma_rayleigh(lam)`（解析，含色散）、`sigma_o3(lam)`（查表插值实测数据）。
- `geometry.py` — 视线几何：给定擦边高度，返回沿视线的 (高度, 弧长) 采样；以及本影几何 `tangent_height(r)` 把月面径向位置映射到擦边高度。
- `radiative_transfer.py` — 核心：组装 τ(λ,h)，算 T(λ,h)、出射谱 I(λ,h)。全程 numpy 矩阵化（λ × h）。
- `color.py` — 颜色科学：CIE CMF 加载、谱→XYZ→sRGB、色相角/亮度提取。
- `solar.py` — 入射谱：真实太阳谱加载 + 5772K 黑体 fallback。
- `pipeline.py` — 串起来：扫一组擦边高度，产出色相/亮度曲线（L0）。

## 关键设计决策

**单次散射 + 视线吸收（v1）。** 我们做的是"沿折射进本影的那束光的视线"上的消光：瑞利散射在这里当作**消光项**（把光散射出视线即损失），臭氧当作**吸收项**。这是经典的 limb transmission 近似，足以重现颜色趋势。多次散射（被散射出去的光部分回到视线）留到 L2 之后评估，预计是二阶修正。

**折射几何先简化。** 真实的月食月光经历地球大气折射弯曲（这正是它能进入本影的原因）。v1 的几何先用"切向穿透 + 给定擦边高度"的直线视线近似来算消光谱，折射主要影响 h↔r 的映射关系和总光程。先把 transmission(h) 算对，再校核 h(r) 映射。

**波长网格。** 380–780nm，先 1–2nm 步长（~200–400 点）。臭氧 Chappuis 带宽缓，不需要超细网格。

**性能路径。** L0 的网格小（数百 λ × 数百 h），纯 numpy 即可秒出。L2 渲染（百万像素 × 数百 λ）才需要上 torch+MPS，把 (像素, λ) 做成大张量在 GPU 上批量积分。先不过早优化。

## 数据契约（data/）

- `data/raw/o3_cross_section_*.{txt,csv}` — 臭氧吸收截面 σ(λ)，列：波长(nm), 截面(cm²)。注明来源与温度。
- `data/raw/atmosphere_profile_*.csv` — 高度(km), 空气数密度(cm⁻³), 臭氧数密度(cm⁻³ 或 VMR)。
- `data/raw/solar_spectrum_*.csv` — 波长(nm), 辐照度。可选，缺则用黑体。
- `data/raw/cie_cmf_1931.csv` — CIE 1931 2° 色匹配函数 x̄ȳz̄。
- 每个原始文件配 `data/raw/SOURCES.md` 记录出处、版本、单位、引用。

## 验证钩子

- `pipeline.py --self-check`：跑默认参数，打印/画出 hue(h), Y(h)，并断言"低 h 偏红、中 h 转青绿、高 h 趋白、Y 单调增"的定性趋势。这是 L0 的 acceptance test。
