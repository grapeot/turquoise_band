# turquoise_band — 月全食绿松石带的物理重现

月全食时，月面从本影深处的暗红、过渡到一条青绿色窄带（**绿松石带 / turquoise band**），再到边缘的正常月光白。本项目从辐射传输第一性出发，定量重现这条带，最终渲染合成月食照片。

物理直觉：折射进地球本影的阳光掠过大气 limb，沿月面径向对应不同擦边高度。低高度处瑞利散射（∝λ⁻⁴）滤掉蓝光留红 → 血月；平流层高度处臭氧 Chappuis 带（~500–700nm）吃掉橙红 → 绿松石；高高度处光程薄 → 接近原始日光白。

## 状态

- [x] Scaffold（repo / venv / PRD / RFC）
- [ ] Science review
- [ ] 下载权威数据（O₃ 截面、大气廓线、太阳谱、CIE CMF）
- [ ] L0：色相曲线闭环（红→青→白趋势自查）
- [ ] L1：定量校核
- [ ] L2：合成月食照片渲染

## 文档

- `docs/PRD.md` — 现象、目标、成功标准、数据需求
- `docs/RFC.md` — 计算管线技术设计、模块划分、验证钩子

## 环境

```bash
source .venv/bin/activate   # uv venv, Python 3.12
```

机器：Apple M3 Ultra（32 CPU / 80 GPU / 512GB），渲染阶段走 MPS。
