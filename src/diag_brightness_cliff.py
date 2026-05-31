"""判决性诊断：蓝带"看起来宽"是 tone map 提暗部造成的，物理上蓝带因亮度悬崖而窄。

左：为暗部曝光（我们之前版本）→ 蓝带被提亮显宽
右：忠实亮度（保留亮度悬崖）→ 蓝带因暗被亮区盖过，自然窄（符合文献"细窄光带"）
下：沿月盘的 红蓝比 + 真实亮度曲线，显示亮度悬崖在最蓝处
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sys.path.insert(0, os.path.dirname(__file__))
import render_rt, render as R

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

t = render_rt.build_branch_tables(n_h=8000)
d = 47.0
Rm = R.R_MOON_ARCMIN
size = 500


def render_disk(expose_mode):
    half = Rm + 4
    xs = np.linspace(d - half, d + half, size)
    ys = np.linspace(-half, half, size)
    Xw, Yw = np.meshgrid(xs, ys)
    a = np.hypot(Xw, Yw)
    inside = np.hypot(Xw - d, Yw) <= Rm
    XYZ = render_rt.shade(a, t)
    Y = np.maximum(XYZ[:, :, 1], 1e-30)
    if expose_mode == "darks":
        # 为暗部曝光：红核侧标到中调 → 暗蓝区被提亮
        a_near = max(d - Rm, t["a_lo"])
        Yd0 = float(render_rt.shade(np.array([a_near]), t)[0, 1])
        E = R._srgb_inv_gamma(0.20) / Yd0
    else:
        # 忠实亮度：按月盘最亮处标定，暗区保持真暗
        E = 5.0 / Y[inside].max()
    XYZ_e = XYZ * E
    rgb = R._srgb_gamma(np.clip(R._xyz_to_srgb_linear(_logtone(XYZ_e)), 0, 1))
    return (np.clip(rgb, 0, 1) * inside[..., None] * 255).astype(np.uint8)


def _logtone(XYZ):
    Y = np.maximum(XYZ[..., 1], 1e-30)
    cx, cz = XYZ[..., 0] / Y, XYZ[..., 2] / Y
    Ymax = max(Y.max(), 1e-6)
    Yd = 0.92 * np.log2(1 + Y) / np.log2(1 + Ymax)
    return np.stack([cx * Yd, Yd, cz * Yd], axis=-1)


fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1.3], hspace=0.28, wspace=0.1)

ax0 = fig.add_subplot(gs[0, 0])
ax0.imshow(render_disk("darks"), origin="lower"); ax0.axis("off")
ax0.set_title("为暗部曝光（我们之前）\n蓝带被提亮 → 看起来占大半月盘", fontsize=11)

ax1 = fig.add_subplot(gs[0, 1])
ax1.imshow(render_disk("bright"), origin="lower"); ax1.axis("off")
ax1.set_title("忠实亮度（保留亮度悬崖）\n蓝带因暗被亮区盖过 → 细窄(符合文献)", fontsize=11)

# 下：剖面
axp = fig.add_subplot(gs[1, :])
xs = np.linspace(d - Rm, d + Rm, 400)
a = np.abs(xs)
XYZ = render_rt.shade(a, t)
Y = np.maximum(XYZ[:, 1], 1e-30)
rgb = np.array([R._srgb_gamma(np.clip(R._xyz_to_srgb_linear(XYZ[i] / Y[i]), 0, 1)) for i in range(len(a))])
rb = (rgb[:, 0] + 1e-6) / (rgb[:, 2] + 1e-6)
xpos = (xs - (d - Rm)) / (2 * Rm)
axp.plot(xpos, rb, color="purple", lw=2, label="红蓝比 R/B")
axp.axhline(1.0, color="gray", ls="--", lw=1)
axp.set_ylabel("红蓝比 R/B", color="purple"); axp.set_ylim(0, 3)
axp.set_xlabel("月盘水平位置 (0=近本影中心, 1=外)")
axp2 = axp.twinx()
axp2.semilogy(xpos, Y, color="orange", lw=2, label="真实亮度(对数)")
axp2.set_ylabel("真实亮度 Y (对数)", color="orange")
# 标蓝区 + 蓝区亮度
blue = rb < 1.0
if blue.any():
    axp.axvspan(xpos[blue].min(), xpos[blue].max(), color="cyan", alpha=0.2)
    axp.text(0.5, 2.6, f"蓝区(R/B<1) {(xpos[blue].max()-xpos[blue].min())*100:.0f}%月盘\n但亮度只有最亮处{Y[blue].max()/Y.max()*100:.0f}%→真实显示下被亮区盖过",
             ha="center", fontsize=9, color="teal")
axp.set_title("亮度悬崖：最蓝处(R/B最低)恰好亮度也很低，旁边趋白区暴亮 → 可见蓝带被压窄", fontsize=10)

fig.savefig(os.path.join(OUT, "moon_brightness_cliff.png"), dpi=140, bbox_inches="tight")
print(f"已存: {os.path.join(OUT,'moon_brightness_cliff.png')}")
