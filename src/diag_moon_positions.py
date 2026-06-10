"""【诊断工具】诊断图：月球停在不同位置(不同 d)时，绿松石带怎么横跨月盘。（基于点源 legacy 着色）

坐标系：原点=本影中心(反日轴)。本影边界=41.2'。绿松石带(h12-18km)在 41.5-49.8'。
月球(视半径15.5')可停在任意 d(月心距本影中心)。月球边缘骑上绿松石带环时，
只有靠外那道弧染青绿，靠内仍是红——这是几何真相。
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import render_rt, render as R

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False

t = render_rt.build_branch_tables(n_h=8000)
Rm, Ru = R.R_MOON_ARCMIN, R.R_UMBRA_ARCMIN
TEAL_LO, TEAL_HI = 41.5, 49.8     # 绿松石带角距范围(h12-18km)

# 全局统一曝光（按最深 d 的红核标定，所有子图同曝光可比）
a_ref = max(26 - Rm, t["a_lo"])
Y_dark = float(render_rt.shade(np.array([a_ref]), t)[0, 1])
E = np.clip(R._srgb_inv_gamma(0.30), 1e-4, .999) / max(Y_dark, 1e-12)

ds = [26, 32, 38]
labels = ["d=26' 月盘全在本影内\n(纯红, 够不到绿松石带)",
          "d=32' 月盘外缘探到绿松石带\n(外缘一道青绿弧)",
          "d=38' 月盘大半在边缘\n(青绿弧更明显, 红心偏内)"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
size = 500
for ax, d, lab in zip(axes, ds, labels):
    half = Rm + 12
    xs = np.linspace(d - half, d + half, size)
    ys = np.linspace(-half, half, size)
    Xw, Yw = np.meshgrid(xs, ys)
    a = np.hypot(Xw, Yw)
    inside = np.hypot(Xw - d, Yw) <= Rm
    XYZ = render_rt.shade(a, t)
    rgb = R._srgb_gamma(np.clip(R._xyz_to_srgb_linear(R._tone_map_on_Y(XYZ, E)), 0, 1))
    rgb = rgb * inside[..., None]
    ax.imshow(rgb, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower")
    # 本影边界 + 绿松石带环
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(Ru*np.cos(th), Ru*np.sin(th), ls="--", color="cyan", lw=1, alpha=0.7)
    ax.plot(TEAL_LO*np.cos(th), TEAL_LO*np.sin(th), ls=":", color="lime", lw=1, alpha=0.6)
    ax.plot(TEAL_HI*np.cos(th), TEAL_HI*np.sin(th), ls=":", color="lime", lw=1, alpha=0.6)
    ax.set_title(lab, fontsize=10)
    ax.set_xlabel("距本影中心 (arcmin)")
    ax.set_aspect("equal")
    ax.set_xlim(xs[0], xs[-1]); ax.set_ylim(ys[0], ys[-1])

fig.suptitle("月球停在不同位置时绿松石带如何横跨月盘（青虚线=本影边界41', 绿点线=绿松石带环41.5-49.8'）",
             fontsize=12)
fig.tight_layout()
p = os.path.join(OUT, "moon_positions.png")
fig.savefig(p, dpi=130)
print(f"已存 {p}")
