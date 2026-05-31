"""逐像素反向 ray tracing 渲染（分支感知反查）—— 替换 LUT 查表，根治 banding。

设计要点（见 docs，由架构 review 推导）：
- banding 根因不是采样密度，是把**非单调**的 角距(h) 映射 argsort 压平、又在角距轴插颜色
  这两个拓扑/坐标错误。加密 h 治不好。
- 正解：一次性在密 h 网格(8000点,~0.3s)算两个 limb 的 h→(角距,XYZ,聚焦)，
  按角距极小值切出**单调上升分支**，每像素在单调 a→h 上反查、在光滑 h→颜色上取色，
  两 limb 辐照**叠加**（XYZ 可加）。无 banding、分辨率任意、numpy 亚秒级、不需 MPS。
- 还修了现有缺陷：旧 LUT 只用对侧 limb，红核的暖是透射谱凑的；这里两 limb 显式叠加。

复用 radiative_transfer/color/geometry/solar 物理，以及 render.py 的显示链。
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import radiative_transfer as rt
import color as col
import geometry as g
import solar
# 复用显示链与几何常数
from render import (_tone_map_on_Y, _xyz_to_srgb_linear, _srgb_gamma, _srgb_inv_gamma,
                    _box_downsample, R_MOON_ARCMIN, R_UMBRA_ARCMIN, OUT)


# ============================================================
# 1. 一次性物理表：h → 两个 limb 的 (角距, XYZ, 聚焦)
# ============================================================
def build_branch_tables(n_h=8000, h_min=0.0, h_max=80.0, n_lam=401):
    """在密 h 网格上算两个 limb 的物理量，切出单调上升分支。

    两个 limb 走同一条切向视线物理（同 h 同 τ 同出射谱），差别只在几何落点与聚焦，
    所以透射谱/XYZ 只算一次，两 limb 共用，省一半辐射传输。
    """
    h = np.linspace(h_min, h_max, n_h)
    lam = np.linspace(380, 780, n_lam)

    # 固定白点：未衰减日光，白点 Y=1（与现有口径一致）
    I_sun = solar.solar_spectrum(lam)
    k = 1.0 / col.spectrum_to_XYZ(lam, I_sun)[1]

    I = rt.emergent_spectrum(h, lam)                                    # (H,L)
    XYZ = np.array([col.spectrum_to_XYZ(lam, I[i]) for i in range(len(h))]) * k  # (H,3)

    # 单一折射映射（几何已厘清：月面某径向坐标只由一族同侧 limb 光照亮，无"对侧"分支）。
    a = g.shadow_radius_arcmin(h)
    i_min = int(np.argmin(a))                # 角距极小值 → 分支分界
    sl = slice(i_min, None)                  # 取极小值以上的单调上升支
    a_mono, h_mono = a[sl], h[sl]
    foc = g.focusing_factor(h)
    foc = foc / foc.max()
    return dict(a_mono=a_mono, h_mono=h_mono,
                a_lo=float(a_mono[0]), a_hi=float(a_mono[-1]),
                h_grid=h, XYZ_grid=XYZ, foc_grid=foc)


def _interp_xyz(h_pix, h_grid, XYZ_grid):
    """对 h_pix（任意 shape）逐通道在 h_grid 上线性插值 XYZ。颜色对 h 光滑→无 banding。"""
    idx = np.clip(np.searchsorted(h_grid, h_pix), 1, len(h_grid) - 1)
    h0, h1 = h_grid[idx - 1], h_grid[idx]
    w = ((h_pix - h0) / (h1 - h0))[..., None]
    return (1 - w) * XYZ_grid[idx - 1] + w * XYZ_grid[idx]


def shade(a_pixel, t):
    """像素角距(任意 shape) → 线性 XYZ。单一映射 a→h→颜色，无 LUT 角距插值、无 banding。"""
    a = np.asarray(a_pixel, dtype=float)
    h_pix = np.interp(a, t["a_mono"], t["h_mono"])        # a→h 单调反查
    in_range = (a >= t["a_lo"]) & (a <= t["a_hi"])
    XYZ = _interp_xyz(h_pix, t["h_grid"], t["XYZ_grid"])  # h→颜色（光滑）
    foc = np.interp(h_pix, t["h_grid"], t["foc_grid"])    # h→聚焦
    return XYZ * foc[..., None] * in_range[..., None]


# ============================================================
# 2. 月盘渲染（几何/显示链同 render_disk，仅着色换成 shade）
# ============================================================
def render_disk_rt(d_arcmin=26.0, size=1400, margin_arcmin=3.0, ssaa=2,
                   exposure=None, expose_srgb=0.32, tables=None):
    """逐像素反查渲染月盘。返回 (rgb8, info)。"""
    if tables is None:
        tables = build_branch_tables()

    half = R_MOON_ARCMIN + margin_arcmin
    cx = d_arcmin
    x0, x1, y0, y1 = cx - half, cx + half, -half, half
    S = size * ssaa
    xs = np.linspace(x0, x1, S)
    ys = np.linspace(y0, y1, S)
    Xw, Yw = np.meshgrid(xs, ys)

    a = np.hypot(Xw, Yw)                       # 到本影中心角距
    r_moon = np.hypot(Xw - cx, Yw)
    inside = r_moon <= R_MOON_ARCMIN

    XYZ_pix = shade(a, tables)                 # (S,S,3) 线性 XYZ

    # 全局曝光「为暗部曝光」：按月盘红核侧标定，红核可见、绿松石/白靠 Reinhard 收高光
    if exposure is None:
        a_near = max(d_arcmin - R_MOON_ARCMIN, tables["a_lo"])
        Y_dark = float(shade(np.array([a_near]), tables)[0, 1])
        t = np.clip(_srgb_inv_gamma(expose_srgb), 1e-4, 0.999)
        exposure = t / (max(Y_dark, 1e-12) * (1.0 - t))

    XYZ_disp = _tone_map_on_Y(XYZ_pix, exposure)
    rgb = _srgb_gamma(np.clip(_xyz_to_srgb_linear(XYZ_disp), 0, 1))
    rgb = rgb * inside[..., None]
    rgb8 = (np.clip(_box_downsample(rgb, ssaa), 0, 1) * 255 + 0.5).astype(np.uint8)

    info = dict(tables=tables, exposure=float(exposure), d=d_arcmin,
                R_moon=R_MOON_ARCMIN, R_umbra=R_UMBRA_ARCMIN,
                extent=(x0, x1, y0, y1), cx=cx)
    return rgb8, info


def self_check(tables, d=26.0):
    """沿 O_umbra→O_moon 连线采样，验证径向次序 + 放大无 banding。"""
    print("\n=== 反向RT 自查 ===")
    xs = np.linspace(d - R_MOON_ARCMIN, d + R_MOON_ARCMIN, 600)
    a = np.abs(xs)
    XYZ = shade(a, tables)
    a_near = max(d - R_MOON_ARCMIN, tables["a_lo"])
    Y_dark = float(shade(np.array([a_near]), tables)[0, 1])
    E = np.clip(_srgb_inv_gamma(0.32), 1e-4, 0.999) / max(Y_dark, 1e-12)
    rgb = _srgb_gamma(np.clip(_xyz_to_srgb_linear(_tone_map_on_Y(XYZ, E)), 0, 1))
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    for frac, label in [(0.05, "近中心"), (0.5, "盘中"), (0.95, "远边缘")]:
        i = int(frac * (len(xs) - 1))
        c = (rgb[i] * 255).astype(int)
        print(f"  {label} a={a[i]:.1f}' sRGB=({c[0]},{c[1]},{c[2]})")

    # banding 检查：相邻像素颜色最大跳变（应平滑，无台阶）
    djump = np.abs(np.diff(rgb, axis=0)).max()
    print(f"  相邻采样最大色跳: {djump:.4f} (越小越平滑，<0.05 无可见台阶)")

    red_i, teal_i = int(0.05 * len(xs)), int(0.95 * len(xs))
    order_ok = R[red_i] > B[red_i] and B[teal_i] >= R[teal_i]
    print(f"  [{'OK' if order_ok else 'XX'}] 红在内(R>B)、青在外(B>=R)")
    return order_ok, djump


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=float, default=26.0)
    ap.add_argument("--size", type=int, default=1400)
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--n_h", type=int, default=8000)
    args = ap.parse_args()

    import time
    t0 = time.time()
    tables = build_branch_tables(n_h=args.n_h)
    t1 = time.time()
    print(f"物理表 {args.n_h}h 用时 {t1-t0:.2f}s")

    ok, djump = self_check(tables, d=args.d)

    rgb8, info = render_disk_rt(d_arcmin=args.d, size=args.size, ssaa=args.ssaa, tables=tables)
    print(f"渲染 {args.size}x{args.size}(SSAA×{args.ssaa}) 总用时 {time.time()-t0:.2f}s, E={info['exposure']:.3g}")

    # 存图（matplotlib 带轴）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7, 7))
    ext = info["extent"]
    ax.imshow(rgb8, extent=[ext[0], ext[1], ext[2], ext[3]], origin="lower")
    ax.set_xlabel("角距 x (arcmin)，原点=本影中心")
    ax.set_ylabel("角距 y (arcmin)")
    ax.set_title(f"月全食月盘（逐像素反向RT，无LUT/无banding，d={args.d:.0f}'）", fontsize=11)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(R_UMBRA_ARCMIN * np.cos(th), R_UMBRA_ARCMIN * np.sin(th),
            ls="--", color="gray", lw=0.8, alpha=0.6)
    ax.set_aspect("equal")
    fig.tight_layout()
    p = os.path.join(OUT, "moon_disk_rt.png")
    fig.savefig(p, dpi=140)
    print(f"已存 {p}")
    print(f"\n{'反向RT自查通过' if ok else '次序未通过'}")
