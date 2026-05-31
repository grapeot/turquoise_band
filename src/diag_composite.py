"""Composite 诊断图：上=月盘渲染，下=沿月盘水平中线的剖面曲线。

剖面横轴 = 月盘上的水平位置(对齐上图)，纵轴展示：
  - 距本影中心角距(arcmin)
  - 红蓝比 R/B（>1 暖, <1 冷）
  - "蓝度" b* (CIELAB, 负=蓝)
让用户看清从哪到哪是绿松石带、占月盘多宽、那片蓝是真青绿还是趋白。
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sys.path.insert(0, os.path.dirname(__file__))
import render_textured as RT
import render_rt, render as R
import colour

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

D = 47.0
# 上图：渲染月盘
rgb8, info = RT.render_realistic_disk(d_arcmin=D, size=1000, ssaa=2)

# 下图：沿月盘水平中线(y=0)采样
t = render_rt.build_branch_tables(n_h=8000)
Rm = R.R_MOON_ARCMIN
xs_world = np.linspace(D - Rm, D + Rm, 600)     # 月盘水平直径
a_line = np.abs(xs_world)                         # 距本影中心角距
XYZ = render_rt.shade(a_line, t)
Y = np.maximum(XYZ[:, 1], 1e-30)
# 颜色(自身亮度归一,看色相) + 红蓝比 + b*
rgb = np.array([R._srgb_gamma(np.clip(R._xyz_to_srgb_linear(XYZ[i] / Y[i]), 0, 1)) for i in range(len(a_line))])
rb = (rgb[:, 0] + 1e-6) / (rgb[:, 2] + 1e-6)
# CIELAB b* (蓝度), 用日光白点
white = t["white"] / max(t["white"][1], 1e-9)
Lab = np.array([colour.XYZ_to_Lab(XYZ[i] / Y[i]) for i in range(len(a_line))])
bstar = Lab[:, 2]

# 蓝带是渐变，"多宽"取决于阈值（这正是肉眼1/5 vs 文献1-2'的差异来源）：
#   - 整片偏冷渐变区（肉眼看成"蓝带"）：红蓝比 < 1（任何蓝多于红）
#   - 文献窄蓝带核心（卫星红/蓝比值成像口径）：红蓝比 < 0.88（最蓝那一丝）
cold_zone = rb < 1.0                           # 肉眼看成蓝的整片冷调渐变区
teal = rb < 0.88                               # 文献口径的窄蓝带核心

# 月盘归一化位置 [0,1]（左=近本影中心红, 右=外侧）
xpos = (xs_world - (D - Rm)) / (2 * Rm)

fig = plt.figure(figsize=(9, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1.4], hspace=0.18)

# --- 上：月盘 ---
ax0 = fig.add_subplot(gs[0])
ax0.imshow(rgb8, origin="lower", extent=[0, 1, 0, 1])
ax0.axhline(0.5, color="yellow", ls=":", lw=0.8, alpha=0.5)   # 采样中线
ax0.set_xticks([]); ax0.set_yticks([])
ax0.set_title(f"月全食月盘 (d={D:.0f}', 蓝带交界过中心)  黄虚线=下方剖面采样位置", fontsize=11)

# --- 下：剖面曲线 ---
ax1 = fig.add_subplot(gs[1])
ax1.plot(xpos, rb, color="purple", lw=2, label="红蓝比 R/B")
ax1.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.6)
ax1.set_ylabel("红蓝比 R/B\n(>1暖红, <1冷蓝)", color="purple")
ax1.set_ylim(0, 3)
ax1.set_xlabel("月盘水平位置 (0=近本影中心/红, 1=外侧)  ——  与上图对齐")
# 标出两个区间：宽的"偏冷渐变区"(肉眼看成蓝) vs 窄的"真蓝带核心"(红蓝比<1)
if cold_zone.any():
    xc = xpos[cold_zone]; wc = a_line[cold_zone].max() - a_line[cold_zone].min()
    ax1.axvspan(xc.min(), xc.max(), color="steelblue", alpha=0.10,
                label=f"偏冷渐变区 {(xc.max()-xc.min())*100:.0f}% / {wc:.0f}' (肉眼看成蓝带)")
if teal.any():
    xt = xpos[teal]; w_arc = a_line[teal].max() - a_line[teal].min()
    ax1.axvspan(xt.min(), xt.max(), color="cyan", alpha=0.35,
                label=f"窄蓝带核心 {(xt.max()-xt.min())*100:.0f}% / {w_arc:.1f}' (文献口径,与Wang2024一致)")
ax1.annotate(f"最蓝 R/B={rb.min():.2f}\n(淡蓝,非饱和)", xy=(xpos[np.argmin(rb)], rb.min()),
             xytext=(0.6, 1.9), fontsize=9, color="navy",
             arrowprops=dict(arrowstyle="->", color="navy", lw=0.8))
ax1.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
# 副轴: b* 蓝度
ax2 = ax1.twinx()
ax2.plot(xpos, bstar, color="blue", lw=1.2, ls="-.", alpha=0.6, label="b* (负=蓝)")
ax2.axhline(0, color="blue", ls=":", lw=0.6, alpha=0.4)
ax2.set_ylabel("CIELAB b* (负=蓝)", color="blue")

# 顶部贴一条实际颜色条带, 对齐 xpos
ax1b = ax1.inset_axes([0, 1.02, 1, 0.08])
ax1b.imshow(rgb[None, :, :], aspect="auto", extent=[0, 1, 0, 1])
ax1b.set_xticks([]); ax1b.set_yticks([])

fig.savefig(os.path.join(OUT, "moon_composite.png"), dpi=140, bbox_inches="tight")
print(f"已存 composite: {os.path.join(OUT,'moon_composite.png')}")
print(f"冷色带(红蓝比<1非白)占月盘宽度: {(xpos[teal].max()-xpos[teal].min())*100:.0f}%" if teal.any() else "无冷色带")
print(f"  角距范围 {a_line[teal].min():.1f}-{a_line[teal].max():.1f}arcmin" if teal.any() else "")
