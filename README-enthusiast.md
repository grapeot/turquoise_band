# turquoise_band — 用代码重现月全食的绿松石带

月全食的时候，月亮进入地球影子后并不会全黑，而是呈现出一系列颜色：影子深处是暗红色，往外会经过一道窄窄的青绿色带（也就是绿松石带 / turquoise band），再往外恢复到正常的月光白。本项目从大气中光传播的基本物理规律出发，逐步建立模型把这些颜色定量算出来，最终合成一张跟真实月食一样的照片。

**站点（图片、视频和六步推演过程）**：https://grapeot.github.io/turquoise_band/ ｜ 物理原理的详细解读见站点 principles 页面。

背后的物理直觉是这样：阳光穿过地球大气往月球方向折射时，每道光掠过的离地高度不同——有的擦过大气底层，有的走平流层，有的从高层穿过。这些高度差异决定了光的颜色。

在大气底层，空气分子对短波长光（蓝光）的散射远强于长波长光（红光）——这就是瑞利散射，也是为什么晴天天空是蓝的、日落是红的。蓝光被散射掉以后，剩下来的红光照到月面上，形成血月。到了平流层高度，臭氧在 500–700nm 有一个吸收带（叫做 Chappuis 带），会把橙光和红光吸收掉。红光被吃掉后，透过来的光就偏青绿色了，绿松石带的颜色就是这么来的。再往高处走，光穿过的大气很薄，吸收和散射都很少，颜色接近日光本身的白色。

## 状态

- [x] **L0 辐射传输闭环**：色相曲线红→青→白，实测数据（O₃ Serdyuchenko 2014、AFGL 大气、SAO2010 太阳谱、CIE CMF）
- [x] **L1 折射几何（解析）**：单一映射 `r(h)=(R⊕+h)−α·d_moon`，对照 Mallama Table 3.1 中高 h 吻合<2%（低 h 端 ~6-10%）
- [x] **L2 逐像素反向 RT 渲染**（render_rt.py）：分支感知反查，无 banding，numpy 亚秒级
- [x] **L3 写实月盘**：NASA 月面纹理 × 物理食光颜色，对数 tone map，忠实亮度（蓝带物理真实地窄）
- [x] **太阳圆盘 ray tracing**（brute_ray_trace.py）：32' 太阳圆盘比带宽 5 倍，把点源版过饱和的浓青(0.41)柔化到符合观测的浅青(0.70)
- [x] **真·正向 ray tracing**（raytrace_eclipse.py，当前权威管线）：真折射 RK4 + 弯曲路径消光（用真实切点高度）+ 撒线 focusing 涌现 + 半影直射光，零解析处方。本影中心 ≈−13.5 档
- [x] **HDR gain map**：可下 iPhone Photos 的 HEIC（tools/make_gainmap_hdr.swift）

### 核心物理结论

绿松石带的颜色本质是青色偏蓝绿的、淡的、窄的。沿着月盘的径向看，颜色（红蓝比 R/B）是平滑渐变的，但你真正能看到的宽度很有限——问题出在亮度上。最蓝的位置比满月暗大约 10 档，紧挨着的趋白区域却亮得多，视觉上只有最靠近蓝带那一条细线能被分辨出来。当前模型算出的最蓝处红蓝比约 0.70（距月心 41'，raw sRGB 口径）。按照 Shu 2024 论文的 654nm/491nm 卫星窄带口径折算后，in-band 红蓝比落在 0.86–1.0 之间，与其实测值 0.8–1.0 重叠（注意两种口径不能直接对比，详见 working.md 续16 与 2026-06-09 勘误）。网络上那些"半个圆盘都是饱和蓝色"的月食图是后期处理的产物。

本影中心偏暗的偏差已定位，源于消光标定的口径差异，见 working.md 2026-06-09/06-10 的相关分析。

### 科学边界

食光的颜色和亮度由大气辐射传输决定，屏幕上看起来什么样由曝光和 tone map 决定——这是两套独立的体系。本项目曾两次差点因为显示效果而误判物理结论（详见 working.md Lessons）。模型的适用范围：

- 本影中心 −15.1 档对应的是最晴夜、无火山灰气溶胶条件下的结果。分子大气不考虑任何气溶胶的理论上限约 −13.5 档。现实中一次具体的月食，如果平流层当时有火山灰或其他气溶胶，本影可以暗得多。**这个模型不能告诉你下一次月食是亮是暗——亮暗由当时的平流层气溶胶状况决定。**
- Shu 2024 卫星窄带口径下 R/B<1 ribbon 在一维径向平均中无法复现，涉及红肩消光和口径差异，为开放项，详见 docs/MODEL_CARD.md。
- I 波段缺乏水汽和氧气吸收带的数据，不引用该波段的绝对亮度值。

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
