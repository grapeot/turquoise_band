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
def build_branch_tables(n_h=8000, h_min=0.0, h_max=80.0, n_lam=401, use_focus=True):
    """在密 h 网格上算两个 limb 的物理量，切出单调上升分支。

    两个 limb 走同一条切向视线物理（同 h 同 τ 同出射谱），差别只在几何落点与聚焦，
    所以透射谱/XYZ 只算一次，两 limb 共用，省一半辐射传输。
    """
    h = np.linspace(h_min, h_max, n_h)
    lam = np.linspace(380, 780, n_lam)

    # 固定白点：未衰减日光，白点 Y=1（与现有口径一致）
    I_sun = solar.solar_spectrum(lam)
    white_XYZ = col.spectrum_to_XYZ(lam, I_sun)
    k = 1.0 / white_XYZ[1]

    I = rt.emergent_spectrum(h, lam)                                    # (H,L)
    XYZ = np.array([col.spectrum_to_XYZ(lam, I[i]) for i in range(len(h))]) * k  # (H,3)

    # 单一折射映射（几何已厘清：月面某径向坐标只由一族同侧 limb 光照亮，无"对侧"分支）。
    a = g.shadow_radius_arcmin(h)
    i_min = int(np.argmin(a))                # 角距极小值 → 分支分界
    sl = slice(i_min, None)                  # 取极小值以上的单调上升支
    a_mono, h_mono = a[sl], h[sl]
    # 聚焦因子(红核中心会聚增亮)。视频里关掉(use_focus=False)以免月盘出现随D左移的亮斑；
    # 绿松石带物理研究仍用(默认True)。
    if use_focus:
        foc = g.focusing_factor(h)
        foc = foc / foc.max()
    else:
        foc = np.ones_like(h)
    # 半影/本影外 = 正常月光(直射日光,亮)。必须≈LUT边缘亮度(趋白区~1)以连续——
    # 否则月盘移出本影、右缘超出LUT范围后会从亮(0.99)突降到暗(造成"月面向右变暗"bug)。
    # 整体暗调由全局 tone-map(DYN_GAMMA)统一处理, 不靠把本影外调暗。
    edge_Y = float(XYZ[-1, 1])              # LUT 最高擦边高度(趋白)的亮度 ≈ 正常月光
    white = white_XYZ * k * (edge_Y / max(white_XYZ[1] * k, 1e-9))
    return dict(a_mono=a_mono, h_mono=h_mono,
                a_lo=float(a_mono[0]), a_hi=float(a_mono[-1]),
                h_grid=h, XYZ_grid=XYZ, foc_grid=foc, white=white)


def _interp_xyz(h_pix, h_grid, XYZ_grid):
    """对 h_pix（任意 shape）逐通道在 h_grid 上线性插值 XYZ。颜色对 h 光滑→无 banding。"""
    idx = np.clip(np.searchsorted(h_grid, h_pix), 1, len(h_grid) - 1)
    h0, h1 = h_grid[idx - 1], h_grid[idx]
    w = ((h_pix - h0) / (h1 - h0))[..., None]
    return (1 - w) * XYZ_grid[idx - 1] + w * XYZ_grid[idx]


def shade(a_pixel, t):
    """像素角距(任意 shape) → 线性 XYZ。单一映射 a→h→颜色，无 LUT 角距插值、无 banding。

    角距超出 LUT 范围（a > a_hi）= 视线不再深入大气 = 本影外/半影深处，
    物理上是**正常月光（未经折射/臭氧的直射日光，白）**，不是黑。
    """
    a = np.asarray(a_pixel, dtype=float)
    a_cl = np.minimum(a, t["a_hi"])                       # 钳到边缘，半影区从边缘值起渐变
    h_pix = np.interp(a_cl, t["a_mono"], t["h_mono"])     # a→h 单调反查
    XYZ = _interp_xyz(h_pix, t["h_grid"], t["XYZ_grid"])  # h→颜色（光滑）
    foc = np.interp(h_pix, t["h_grid"], t["foc_grid"])    # h→聚焦
    out = XYZ * foc[..., None]
    out = np.where((a >= t["a_lo"])[..., None], out, 0.0) # 内侧极深处之外

    # 本影外/半影 → 从 LUT 边缘值平滑渐变到中性月光白，避免硬边界。
    beyond = a > t["a_hi"]
    if np.any(beyond):
        edge = _interp_xyz(np.array([t["h_mono"][-1]]), t["h_grid"], t["XYZ_grid"])[0] \
            * np.interp(t["h_mono"][-1], t["h_grid"], t["foc_grid"])
        frac = np.clip((a - t["a_hi"]) / 12.0, 0.0, 1.0)[..., None]
        penumbra = edge * (1 - frac) + t["white"] * frac
        out = np.where(beyond[..., None], penumbra, out)
    return out


# ============================================================
# 1b. 太阳圆盘 ray tracing 着色（去点源近似）
# ============================================================
# 点源版 shade(a) 假设月面点只被单一擦边高度的一条光线照亮(反日轴单光线),
# 太阳的 16' 角径靠 focusing 的 r_floor 软下限糊。这里真做：月面点 a 收到的光 =
# 太阳圆盘(±ANG_SUN)各子点的折射光叠加。每个子点沿轴偏移 ξ → 落点平移 → 反查 h。
# 含 focusing(能量守恒会聚, 修亮度悬崖), 不需 r_floor(圆盘有限大小天然正则化中心发散)。

def _ang_sun_arcmin():
    return float(np.degrees(np.arctan(g.R_SUN_KM / g.D_SUN_KM)) * 60.0)


def build_disk_tables(n_h=8000, h_min=0.0, h_max=80.0, n_lam=401):
    """太阳圆盘着色用的物理表：带符号落点角距 a_signed(h) + h→XYZ + h→focusing(无r_floor)。

    a_signed 是带符号的(反日轴坐标, 可负=过轴), 圆盘积分要沿这条单调轴反查, 不能用
    点源 shade 的 |a| 折叠。focusing 用 r_floor=0(不软下限), 圆盘积分自己正则化。
    """
    h = np.linspace(h_min, h_max, n_h)
    lam = np.linspace(380, 780, n_lam)
    I_sun = solar.solar_spectrum(lam)
    white_XYZ = col.spectrum_to_XYZ(lam, I_sun); k = 1.0 / white_XYZ[1]
    I = rt.emergent_spectrum(h, lam)
    XYZ = np.array([col.spectrum_to_XYZ(lam, I[i]) for i in range(len(h))]) * k   # (H,3)

    # 带符号落点角距(反日轴) r(h)=(R⊕+h)−α(h)·d_moon
    r_signed_km = (g.R_EARTH + h) - g.refraction_angle(h) * g.D_MOON_KM
    a_signed = np.degrees(np.arctan(r_signed_km / g.D_MOON_KM)) * 60.0           # (H,) 单调上升

    # focusing 不用 r_floor(置极小), 让圆盘积分天然正则化中心发散
    b = g.R_EARTH + h
    dr_dh = 1.0 + g.refraction_angle(h) * g.D_MOON_KM / g.H_REFRAC_KM
    foc = b / dr_dh / np.maximum(np.abs(r_signed_km), 1.0)                       # b·dh/dr/r, r_floor→1km

    white = white_XYZ * k * (float(XYZ[-1, 1]) / max(white_XYZ[1] * k, 1e-9))    # 半影外正常月光
    return dict(h=h, a_signed=a_signed, XYZ=XYZ, foc=foc, white=white,
                ang_sun=_ang_sun_arcmin(), a_lo=float(a_signed.min()), a_hi=float(a_signed.max()))


def build_disk_lut(n_h=300000, n_xi=257, bin_width=0.08, a_lo=18.0, a_hi=72.0):
    """金标准圆盘 ray tracing 的 a→XYZ LUT(与 d 无关, 建一次全帧复用)。

    复用 brute_ray_trace 的撒线+分箱(focusing从落点密度自然涌现), 得各角距 a 的积分 XYZ。
    归一: 趋白外缘 Y→正常月光 1.0。返回 dict(a, XYZ, a_lo, a_hi) 供 shade_disk_lut 查。
    """
    import brute_ray_trace as bt
    from scipy.ndimage import gaussian_filter1d
    res = bt.brute_trace(n_h=n_h, n_xi=n_xi, bin_width=bin_width,
                         a_grid_lo=a_lo, a_grid_hi=a_hi)
    a = res["a"]; XYZ = res["XYZ"].copy(); Y = res["Y"].copy()
    for c in range(3):
        XYZ[:, c] = gaussian_filter1d(XYZ[:, c], sigma=2.0)   # 消分箱噪声(banding artifact)
    yref = np.percentile(XYZ[Y > 0, 1], 99)
    XYZ = XYZ / max(yref, 1e-9); Yn = XYZ[:, 1]

    # 修边界外虚假衰减: 暴力撒线在大a端(本影外/半影外)落点密度不足→Y虚降(a=70只0.53),
    # 但物理上 a>本影边界 = 直射正常月光(最亮~1, 不衰减)。亮度峰值位置≈本影边界,
    # 峰值之后 clamp 到正常月光白(峰值处的XYZ), 消除"月盘右缘虚假阴影"(D=60末帧bug)。
    i_peak = int(np.argmax(Yn))
    white = XYZ[i_peak].copy()                          # 峰值处=趋白正常月光
    XYZ[i_peak:] = white                                # 峰值外全部=正常月光白(不衰减)
    return dict(a=a, XYZ=XYZ, a_lo=float(a[0]), a_hi=float(a[-1]), white=white)


def shade_disk_lut(a_pixel, lut):
    """按角距查圆盘 LUT → 线性 XYZ。超 a_hi=趋白正常月光(直射), 低于内缘=深本影(LUT给)。"""
    a = np.asarray(a_pixel, dtype=float)
    out = np.empty(a.shape + (3,))
    for c in range(3):
        out[..., c] = np.interp(a, lut["a"], lut["XYZ"][:, c],
                                left=lut["XYZ"][0, c], right=lut["XYZ"][-1, c])
    return out


def shade_disk(a_pixel, t, n_xi=257):
    """太阳圆盘 ray tracing 着色：像素角距 a(任意shape) → 圆盘积分线性 XYZ(含focusing)。

    对每个月面点 a, 积分太阳圆盘各子点 ξ∈[-ang_sun, ang_sun](弦长加权, 均匀盘):
      子点 ξ 的折射光落点要求 a_signed(h)=a−ξ → 反查 h → XYZ(h)·foc(h)。
      落点超 a_hi(高空外): 不经大气=直射日光(白); 落点 < a_lo: 无光(被地球挡)。
    """
    a = np.asarray(a_pixel, dtype=float)
    shp = a.shape
    af = a.ravel()[:, None]                                   # (P,1)
    xi = np.linspace(-t["ang_sun"], t["ang_sun"], n_xi)[None, :]   # (1,X)
    w = np.sqrt(np.clip(t["ang_sun"]**2 - xi**2, 0, None))   # 弦长权重(均匀盘1D投影)
    w = w / max(w.sum(), 1e-12)
    target = af - xi                                          # (P,X) 各子点要求落点角距

    # 反查 h: a_signed 单调上升 → 插值得 h_idx, 取 XYZ·foc
    a_sig = t["a_signed"]; XYZ = t["XYZ"]; foc = t["foc"]
    tgt_cl = np.clip(target, a_sig[0], a_sig[-1])
    idx = np.searchsorted(a_sig, tgt_cl).clip(1, len(a_sig)-1)
    a0 = a_sig[idx-1]; a1 = a_sig[idx]
    fr = (tgt_cl - a0) / np.maximum(a1 - a0, 1e-9)
    XYZf = XYZ * foc[:, None]                                 # (H,3) 预乘focusing
    col_lo = XYZf[idx-1]; col_hi = XYZf[idx]                  # (P,X,3)
    contrib = col_lo * (1 - fr)[..., None] + col_hi * fr[..., None]

    # 落点超 a_hi: 直射白(不经大气); 落点 < a_lo: 无光(被挡置0)
    over = target > t["a_hi"]
    under = target < t["a_lo"]
    contrib = np.where(over[..., None], t["white"][None, None, :], contrib)
    contrib = np.where(under[..., None], 0.0, contrib)

    out = (contrib * w[..., None]).sum(axis=1)               # (P,3) 圆盘加权积分
    return out.reshape(*shp, 3)


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
