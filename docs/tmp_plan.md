# tmp_plan — 收尾四件事

视频已重渲上传（SDR/HDR，无 floor 三 panel 独立 tone map）。剩下四件，一件件来，自己定节奏。

## 状态
- [x] 1. 重渲 SDR+HDR 上传（无floor三panel独立tonemap，60fps 300帧）
- [x] 2. reassess 绿松石带——**无回归**，物理仍符合文献(R/B 0.88)，亮度悬崖更强(蓝带窄结论更稳)。见 working.md 续7
- [ ] 3. 逐因素叠加渲染 demo（黑白→消光→瑞利→臭氧→太阳谱）
- [ ] 4. 整理文档 + 删临时文件 + commit

---

## 第2件：reassess 绿松石带 against literature

**为什么要做**：这一长轮改了几个会影响 render_rt/render_textured 的东西，要验证绿松石带的物理结论还成立：
- 关掉聚焦因子（use_focus，但那是 build_video 的 MOON_T，render_textured 默认仍 use_focus=True——要确认）
- 半影白点改成正常月光（连续，之前 0.20→正常月光）
- limb darkening、tone map 改动（这些在 build_video，不影响 render_textured）

**怎么做**：跑 `python src/pipeline.py --self-check` 和 `python src/render_textured.py`，看：
- 绿松石带红蓝比是否仍 ~0.88（文献 0.8-1.0）
- 严格口径带宽是否仍 ~2 arcmin / 8%（文献 Shu2024 1-2'）
- 色相、角位置是否仍与 Mallama/Shu 吻合
- 对比 docs/working.md 续4 记录的结论，看有无 regression

**关键检查点**：聚焦因子默认是否影响了 render_textured（它只乘亮度不碰色相，理论上不影响颜色结论，但要验证带宽/亮度悬崖结论没变）。半影白点改连续后，亮度悬崖（蓝带窄的根因）是否还在。

**产出**：在 working.md 加一条 reassess 结论（物理仍符合 / 哪里变了）。

---

## 第3件：逐因素叠加渲染 demo

**目标**：教学/解释性。回到绿松石带那张图（月盘中心正好在最蓝处，d 让中心对最蓝角距）。
从黑白开始，一个物理因素一个加进来，每步一张图，展示每个机制贡献了什么。

**因素序列**（用户定）：
1. **黑白**：禁用所有物理。有光=白（直射日光），无光=黑（被地球完全挡）。月盘=纯黑（全在本影）或纯白。
2. **+大气消光**：加 Beer-Lambert 总消光（瑞利+臭氧合一，或灰消光）→ 本影内不再全黑，中间灰。
3. **+瑞利散射**：瑞利 ∝λ⁻⁴ 单独 → 暗红出来（血月，蓝光散射掉留红）。
4. **+臭氧 Chappuis**：加臭氧吸收 → 绿松石带出现（吃600nm橙红，留蓝绿）。
5. **+实测太阳谱**：黑体→SAO2010 → 最终版（色相微调）。

**框架设计**（待定，需想清楚）：
- 这几个因素彼此依赖（瑞利和臭氧都是消光的组成；太阳谱是入射）。"黑白"和"灰消光"是简化的教学态，不是物理管线的真实中间态。
- 可能要给 radiative_transfer / cross_sections 加开关（enable_rayleigh, enable_ozone, solar=blackbody/real, extinction_mode=none/gray/physical）。
- 一个 Python 脚本 `src/demo_factors.py` 扫这5个配置，各渲一张月盘图（用现成的 render_textured 几何+月盘）。
- 中心对最蓝：d 选让月盘中心角距≈最蓝处(~52')。但月盘半径15.5,中心52则覆盖[36,68]——要确认几何。

**需要用户输入的点**：
- "黑白"态怎么定义（纯几何遮挡？还是消光=∞/0两档？）
- 每张图是月盘（带纹理）还是纯色盘（不带纹理，更干净看物理）？
- 要不要配文字说明（每张图标"+瑞利"等）

**说明怎么写**：用户想想。这是个复杂的彼此依赖+依赖输入的东西，先有图再配文。

---

## 第3.5件：SDR tone mapping 三问题（用户截图 D=44.4' clarification）

**根因：三个问题都是同一个——tone mapping 加得太平（动态范围压缩过头/曲线斜率太低）。**

1. **月球**：高对比帧（右=阳光全照、左=几乎本影），两边亮度该差很多，但现在差不多。
   → tone map 把暗亮压平了。要恢复 spatial 对比：右边正常月光该明显亮、左边本影该明显暗。
2. **地球 panel**：地球夜面太亮，和外面大气色环几乎一样亮。
   → 夜面该明显暗于环。tone map 太平把它们拉到相近亮度。
3. **大气特写（最关键的物理 hint）**：从 ~30 角分到最后再没变过，**而且它还有颜色（没饱和到白）**。
   - 用户洞察：太阳露出/钻石环时大气该 **bloom/saturate 到白**。"它还有颜色 = 它没 saturate"，物理上不对。
   - 即：亮部该有真正的 clip-to-white（bloom），而不是被 tone map 软压住一直保持颜色。
   - 以前的版本记得有这个效果，现在丢了 → tone map 改动把饱和/bloom 压没了。

**方向**：tone map 曲线要更"陡"（spatial 对比回来）+ 高光要能真正 clip 到白（bloom，太阳露出时大气饱和）。
当前 `_panel_tonemap` 的 gamma+对数肩部把：(a) 动态范围压平了对比 (b) 对数肩部让高光永不到白。
可能要：减小 gamma 压缩（更线性保对比）+ 去掉/弱化对数肩部让亮部能 clip 到 1（白）。

## 第4件：整理 + commit
- 删临时诊断文件（outputs/ 下的 test_*/check_*/tuned_*/newtone_*/fix*/nofloor_* 等中间产物）
- 整理 docs（working.md 加 reassess 结论；本 tmp_plan 完成后可删或归档）
- commit

---

## 节奏（用户定的顺序）
1. [x] reassess 绿松石（验证物理没坏）
2. [ ] 落盘 commit（本次）
3. [ ] 写 plan（含三条SDR feedback，本次已写）
4. [ ] 回头更新 SDR（解决 tone map 太平 + 高光不饱和 三问题）→ commit
5. [ ] 多因素 evaluation study + visualization（逐因素叠加）
到那时用户会有新反馈。
