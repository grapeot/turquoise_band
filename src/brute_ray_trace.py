"""暴力 2D ray tracing —— 月食绿松石带的金标准。

完全独立、第一性原理。不查 a_signed 反插值、不用任何 focusing 公式。
直接撒密集平行光线，统计落点密度，让能量守恒/聚焦从落点扎堆自然涌现。

物理图景(2D 子午面, 反日轴为 x 轴):
  - 太阳不是点源, 角半径 ANG_SUN=16'。把太阳圆盘沿径向投影成 1D 弦长加权分布:
    一个子点的角偏移 ξ∈[-16',+16'], 权重 ∝ 弦长 sqrt(R_sun^2-ξ^2)(均匀亮度圆盘的 1D 投影)。
  - 对每个子点 ξ, 它发出一束平行光(方向偏离反日轴 ξ)。这束光里, 每条光线由它擦地球
    +limb 的高度 h 标定: impact parameter b=R⊕+h。
  - 擦地高度 h 的光线被折射 α(h)=α0·exp(-h/8) 偏向轴。其落到月面(反日轴坐标)的
    带符号位置:
        x_land(h, ξ) = a_signed(h) + ξ        (arcmin, 反日轴坐标)
    其中 a_signed(h)=arctan([(R⊕+h)−α(h)·d_moon]/d_moon), 是该子点对应的"轴上落点",
    子点偏移 ξ 把整条折射映射整体平移 ξ(平行光入射方向偏 ξ → 落点偏 ξ)。
  - 这条光线携带出射谱 I_sun(λ)·T(λ,h)(只取决于擦地高度 h, 与 ξ 无关)。
  - 撒法: 在 impact parameter b(等价于 h)上**均匀**撒线(代表均匀照射地球 limb 的平行光通量),
    对每个 ξ 子点撒一遍。把所有光线按落点 x_land 装进角距 bin, 每个 bin 内累加各光线的
    XYZ(=出射谱积分 CMF)。落点扎堆的 bin 自然亮(focusing 涌现), 不手动乘任何因子。

  能量记账(为什么"均匀撒 b"就对): 2D 子午面里, limb 上 [b,b+db] 这一薄环接收的平行光
  通量 ∝ db(平行光均匀)。我们在 b 上均匀撒 N 条线, 每条权重相同 = 该 db 的能量。它们
  折射后落到 [x,x+dx], 落点密度 ∝ db/dx = 1/(dx/db) = focusing。bin 计数自然正比于此。
  这就是 focusing 的第一性来源, 无需 b·|dh/dr|/r 解析式。

  径向 vs 全 2D: 真实是 3D 轴对称。子午面 1D 撒线给出沿反日轴一条直径上的剖面;
  绿松石带是轴对称环, 沿任意半径剖面相同, 故 1D 子午剖面即为答案。太阳圆盘的 2D 面
  用弦长加权投影到径向 ξ(均匀盘 → 半圆分布), 这是把 2D 圆盘卷积正确降到 1D 的标准做法。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import geometry as g
import radiative_transfer as rt
import solar
import color as col

ARCMIN_PER_RAD = np.degrees(1.0) * 60.0


def _n_minus_1(lam_nm):
    """干燥空气折射率 (n-1)，Edlén 1966 简化式。决定折射角的波长依赖(色散)。"""
    sig2 = (1.0e3 / np.asarray(lam_nm, float)) ** 2
    return (8342.54 + 2406147.0 / (130.0 - sig2) + 15998.0 / (38.9 - sig2)) * 1e-8


def dispersion_scale(lam_nm, lam_ref=600.0):
    """折射角随波长的缩放因子 α(λ)/α(ref) = (n(λ)-1)/(n(ref)-1)。
    蓝光(n-1大)折射更强→落点更靠外, 像棱镜。ref=600nm(原消色差α0对应处)。"""
    return _n_minus_1(lam_nm) / _n_minus_1(lam_ref)


def a_signed_arcmin(h_km, disp_scale=1.0):
    """擦地高度 h → 轴上落点带符号角距(arcmin)。点源(轴上子点)版本。

    r_signed = (R⊕+h) − α(h)·d_moon ; a = arctan(r_signed/d_moon)。
    低 h 强折射 → r 负(落本影深处/红核, 过轴); 高 h 弱折射 → r 大(外缘绿松石带)。
    """
    r_signed_km = (g.R_EARTH + h_km) - g.refraction_angle(h_km) * disp_scale * g.D_MOON_KM
    return np.degrees(np.arctan(r_signed_km / g.D_MOON_KM)) * 60.0


def ang_sun_arcmin():
    return float(np.degrees(np.arctan(g.R_SUN_KM / g.D_SUN_KM)) * 60.0)


def brute_trace(
    n_h=120000,        # 擦地高度方向撒线数(沿 impact parameter 均匀)
    h_min=0.0,
    h_max=80.0,
    n_xi=129,          # 太阳圆盘径向子点数
    n_lam=401,
    bin_width=0.25,    # 月面角距 bin 宽(arcmin)
    a_grid_lo=35.0,    # 统计范围(arcmin), 覆盖绿松石带 46-52'
    a_grid_hi=60.0,
    point_source=False,
    n_disp=16,         # 折射色散波段数(蓝光折射更强→落点更靠外)。=1 关色散(ablation 用)
):
    """暴力撒线 + 落点分箱。返回各角距 bin 的 (a, R/B, Y, XYZ)。

    返回 dict: a_centers, RB, Y, XYZ(归一前的累加值), counts。
    """
    lam = np.linspace(380.0, 780.0, n_lam)

    # 白点: 未衰减日光, 用于把 XYZ 标定到 Y(白)=1
    I_sun_full = solar.solar_spectrum(lam)
    white_XYZ = col.spectrum_to_XYZ(lam, I_sun_full)
    k_white = 1.0 / white_XYZ[1]

    # --- 预计算每个擦地高度 h 的出射谱(H,L), 与 ξ 无关一次算好 ---
    # 折射色散: 不同波段用不同折射缩放 → 不同落点。把谱按波长分 n_disp 段,
    # 每段单独 trace 落点(用该段代表波长的色散缩放), 实现"蓝光落点更靠外"。
    n_h_phys = 4000
    h_nodes = np.linspace(h_min, h_max, n_h_phys)
    I_emerg = rt.emergent_spectrum(h_nodes, lam)                      # (H,L) 各h的出射谱

    # 波长分段: 每段一个色散缩放 + 该段对应的 XYZ(只积该段波长的 CMF)
    band_edges = np.linspace(0, len(lam), n_disp + 1).astype(int)
    bands = []
    for bi in range(n_disp):
        sl = slice(band_edges[bi], band_edges[bi + 1])
        if sl.start >= sl.stop:
            continue
        lam_b = lam[sl]
        lam_rep = float(lam_b.mean())
        dsc = dispersion_scale(lam_rep) if n_disp > 1 else 1.0
        # 该段的 XYZ(只积本段波长): 各h节点
        XYZ_b = np.array([col.spectrum_to_XYZ(lam_b, I_emerg[i, sl]) for i in range(n_h_phys)]) * k_white
        bands.append((dsc, XYZ_b))

    # --- 撒线: impact parameter b=R⊕+h 均匀, 等权(等通量) ---
    h_rays = np.linspace(h_min, h_max, n_h)

    # --- 太阳圆盘径向子点 ξ + 弦长权重(均匀亮度盘的 1D 投影) ---
    ang_sun = ang_sun_arcmin()
    if point_source:
        xis = np.array([0.0])
        w_xi = np.array([1.0])
    else:
        xis = np.linspace(-ang_sun, ang_sun, n_xi)
        w_xi = np.sqrt(np.clip(ang_sun**2 - xis**2, 0.0, None))
        w_xi = w_xi / w_xi.sum()

    # --- 落点分箱 ---
    edges = np.arange(a_grid_lo, a_grid_hi + bin_width, bin_width)
    a_centers = 0.5 * (edges[:-1] + edges[1:])
    nb = len(a_centers)
    XYZ_bin = np.zeros((nb, 3))
    cnt_bin = np.zeros(nb)

    # 每条光线代表的"轴通量份额": b 上均匀撒 → 等权; 但要乘 2π·b? 不。
    # 2D 子午剖面: 我们要的是沿反日轴半径方向的 1D 落点线密度, 平行光在 limb 高度 b 上的
    # 线通量 ∝ db(均匀)。所以每条线等权 1。focusing= 落点线密度比, 由分箱涌现。
    base_w = 1.0

    # 每个色散波段: 用该段色散缩放算落点 a_axis, 把该段 XYZ 分箱。
    # 蓝段 dsc>1 → 折射更强 → 落点更靠外(像棱镜把蓝光多弯一点)。色散从这里涌现。
    for dsc, XYZ_b in bands:
        a_axis = a_signed_arcmin(h_rays, disp_scale=dsc)             # (N,) 该波段落点
        XYZ_ray = np.empty((n_h, 3))
        for c in range(3):
            XYZ_ray[:, c] = np.interp(h_rays, h_nodes, XYZ_b[:, c])
        for xi, wj in zip(xis, w_xi):
            x_land = a_axis + xi
            idx = np.floor((x_land - a_grid_lo) / bin_width).astype(int)
            m = (idx >= 0) & (idx < nb)
            ii = idx[m]
            ww = wj * base_w
            np.add.at(XYZ_bin, ii, XYZ_ray[m] * ww)
            np.add.at(cnt_bin, ii, ww * np.ones(m.sum()))

    # bin 的代表 XYZ = 累加值(已含落点密度 → focusing)。Y 直接当亮度。
    # R/B: 由 bin 累加的 XYZ → sRGB 线性 R,B 之比。
    Y = XYZ_bin[:, 1].copy()
    # 归一化 Y 到峰值(相对亮度)
    Y_rel = Y / max(Y.max(), 1e-30)

    # R/B: 把累加 XYZ 转线性 sRGB, 取 R/B
    RB = np.full(nb, np.nan)
    for i in range(nb):
        if XYZ_bin[i, 1] <= 0:
            continue
        rgb_lin = col._xyz_to_srgb_linear_safe(XYZ_bin[i]) if hasattr(col, "_xyz_to_srgb_linear_safe") else _xyz_lin(XYZ_bin[i])
        R, B = rgb_lin[0], rgb_lin[2]
        if B > 1e-30:
            RB[i] = R / B

    return dict(a=a_centers, RB=RB, Y=Y, Y_rel=Y_rel, XYZ=XYZ_bin, counts=cnt_bin,
                ang_sun=ang_sun)


# sRGB 线性变换矩阵(D65), 用于从 XYZ 取线性 R/B
_M_XYZ2RGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])


def _xyz_lin(XYZ):
    return _M_XYZ2RGB @ np.asarray(XYZ, float)


def cliff_width(a, Y_rel):
    """亮度悬崖宽度: Y 从 5%max 到 50%max 的角距跨度(arcmin)。

    绿松石带在亮度悬崖外沿。取 Y 上升沿(从外向内? 实际从本影外亮 → 本影内暗)。
    这里沿角距递增方向(向外): 找 Y_rel 穿过 0.05 和 0.50 的角距, 取其间距。
    """
    # 在统计区内 Y 随 a 先升(进入绿松石带)后降? 实际: 大角距=本影边缘外侧=亮,
    # 小角距=深入本影=暗。悬崖是 Y 从暗(本影深处)快速升到亮(边缘)的过渡带。
    # 沿 a 递增方向找 Y_rel 首次 >=0.05 和首次 >=0.50 的位置。
    a = np.asarray(a); Y = np.asarray(Y_rel)
    def cross(level):
        for i in range(1, len(Y)):
            if Y[i - 1] < level <= Y[i]:
                # 线性插值
                f = (level - Y[i - 1]) / (Y[i] - Y[i - 1])
                return a[i - 1] + f * (a[i] - a[i - 1])
        return np.nan
    a5 = cross(0.05)
    a50 = cross(0.50)
    return abs(a50 - a5), a5, a50


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_h", type=int, default=120000)
    ap.add_argument("--n_xi", type=int, default=129)
    ap.add_argument("--bin", type=float, default=0.25)
    ap.add_argument("--point", action="store_true")
    args = ap.parse_args()

    import time
    t0 = time.time()
    res = brute_trace(n_h=args.n_h, n_xi=args.n_xi, bin_width=args.bin, point_source=args.point)
    dt = time.time() - t0

    a, RB, Y_rel = res["a"], res["RB"], res["Y_rel"]
    # 最蓝(R/B 最小)
    valid = np.isfinite(RB) & (res["Y"] > res["Y"].max() * 1e-3)
    ib = np.nanargmin(np.where(valid, RB, np.inf))
    cw, a5, a50 = cliff_width(a, Y_rel)

    print(f"暴力 ray tracing 完成, n_h={args.n_h} n_xi={args.n_xi} 用时 {dt:.2f}s")
    print(f"太阳角半径 {res['ang_sun']:.2f}'")
    print("\n角距(')   R/B     Y_rel")
    for i in range(len(a)):
        if 40 <= a[i] <= 58:
            print(f"  {a[i]:5.2f}  {RB[i]:6.3f}  {Y_rel[i]:.4f}")
    print(f"\n最蓝 R/B={RB[ib]:.3f} @ {a[ib]:.2f}'")
    print(f"亮度悬崖: Y 5%@{a5:.2f}' → 50%@{a50:.2f}', 宽度 {cw:.2f}'")
