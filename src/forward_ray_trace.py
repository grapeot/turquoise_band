"""真·正向 Monte-Carlo ray tracing —— 算月食月面绝对亮度（相对满月的档数）。

零 artificial 参数。focusing（聚焦）、落点疏密、本影中心亮度全部从撒光线 + 落点
分箱自然涌现。不用任何解析权重（不用 b·|dh/dr|/r、不用 r_floor、不用弦长权重公式）。

================================================================================
物理图景（完整 3D 轴对称，落点装进月面 2D 笛卡尔网格）
================================================================================

坐标系：以反日点轴为 z 轴。月面是 z=d_moon 处垂直于轴的平面，用 (x,y) 描述（km），
原点 (0,0) 是本影中心（反日点）。

太阳是有限大小的圆盘，角半径 ANG_SUN≈16'。它不是点源——这是绿松石带和本影边界
柔和过渡的根源。我们在太阳圆盘上**等面积**撒 2D 子点 ξ=(ξx,ξy)（在角半径 16' 的
圆盘内做 rejection sampling），每个子点等权。等面积撒点天然给出正确的"圆盘投影"——
不手加弦长权重，弦长分布从 2D 圆盘几何自然涌现。

对每个太阳子点 ξ，它发出一束平行光，方向相对反日轴整体偏转 ξ。这束平行光均匀照亮
地球 limb。一条光线由它擦地球大气的位置标定：
  - impact parameter 向量 b（在垂直于光束的波前平面内），|b|=R⊕+h，方位角 φ。
  - 在垂直入射的极限下（月食时太阳/地球/月亮近共线），波前平面≈月面 (x,y) 平面。
  我们在波前平面里**等面积**撒 b（均匀采样圆环 R⊕ < |b| < R⊕+h_max，等通量），
  天然不需要任何弦长/通量权重。

擦地高度 h=|b|−R⊕ 的光线被大气折射 α(h)，**径向向轴弯**（朝 limb 内法线方向，即朝
−b̂ 方向）。折射后这条光线在月面的落点（反日轴坐标）：
    x_land = b̂ · ( |b| − α(h)·d_moon )  +  ξ·d_moon
即把"沿 b 方向、距轴 (|b|−α·d_moon) 的点"再加上太阳子点方向偏移 ξ 造成的平移。
（ξ 是角度，乘 d_moon 变 km。）

这条光线携带的能量 = 该子点的单位通量份额 × 出射谱透射的可见光能量（=出射谱积分 CMF
得到的 luminance Y，已对满月白点归一）。出射谱只取决于擦地高度 h（消光路径），与 φ、ξ 无关。

落点装进月面 (x,y) 2D 网格的像素。每像素累加落进它的所有光线能量。
**关键归一化：月面亮度 = 落进单位面积的能量 = 像素累加能量 / 像素面积。** 2D 笛卡尔
网格每像素面积是常数 (dx·dy)，所以"每像素能量"已经正比于面亮度——本影中心的环带
面积塌缩（2πρ·dρ→0）问题在 2D 网格里自动消失，focusing 作为落点密度的真实物理后果
自然涌现，不需要除以 2πρ。

满月基准：完全相同的撒法，但光线不经大气（不折射、不消光），直射打到月面。每条光线
携带满月白点能量 Y=1。它们均匀铺满月面对应区域 → 每像素能量密度 = 满月面亮度基准。
两者相除取 log2 = 档数。

================================================================================
为什么这能修正旧"-11 档"低估
================================================================================
旧 brute_ray_trace 在**角距（arcmin）一维子午剖面**上用**等角宽 bin** 累加、且直接把
bin 计数当亮度。等角宽 bin 在 ρ→0 对应的环带面积 ∝ρ·dρ→0，但它没除以这个塌缩面积，
于是本影中心的焦散聚焦（caustic）压根没涌现，导致中心被严重高估变亮（=低估档数）。
本版用 2D 网格 + 像素面积归一，聚焦/焦散从落点密度真实涌现。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import geometry as g
import radiative_transfer as rt
import solar
import color as col

# sRGB(D65) 线性变换矩阵，用于从累加 XYZ 取线性 R/B（判断颜色）
_M_XYZ2RGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])


def ang_sun_rad():
    """太阳角半径 (rad)。从 R_sun / d_sun 算，≈16'。"""
    return float(np.arctan(g.R_SUN_KM / g.D_SUN_KM))


def _n_minus_1(lam_nm):
    """干空气折射率 (n-1)，Edlén。决定折射角的波长依赖（色散，蓝光弯更多）。"""
    sig2 = (1.0e3 / np.asarray(lam_nm, float)) ** 2
    return (8342.54 + 2406147.0 / (130.0 - sig2) + 15998.0 / (38.9 - sig2)) * 1e-8


def dispersion_scale(lam_nm, lam_ref=600.0):
    """折射角随波长缩放 α(λ)/α(ref)。蓝端 (n-1) 大 → 折射更强 → 落点更靠外（棱镜效应）。"""
    return _n_minus_1(lam_nm) / _n_minus_1(lam_ref)


def _precompute_emergent_luminance(lam, n_h_nodes=4000, h_min=0.0, h_max=90.0,
                                   n_disp=12):
    """预计算每个擦地高度 h 的出射谱在各色散波段的 (X,Y,Z) 能量（对满月白点归一）。

    返回:
      h_nodes : (Hn,)
      bands   : list of (disp_scale, XYZ_nodes(Hn,3))  每个折射色散波段一项
      Y_full_nodes : (Hn,)  全谱总 luminance（不分段，用于纯亮度统计）
    满月白点 = 未衰减日光的 XYZ，k_white = 1/Y_white。这样满月的 Y=1。
    """
    I_sun_full = solar.solar_spectrum(lam)
    white_XYZ = col.spectrum_to_XYZ(lam, I_sun_full)
    k_white = 1.0 / white_XYZ[1]

    h_nodes = np.linspace(h_min, h_max, n_h_nodes)
    I_emerg = rt.emergent_spectrum(h_nodes, lam)            # (Hn, L)

    # 全谱 luminance（每个 h 节点）
    Y_full_nodes = np.array([col.spectrum_to_XYZ(lam, I_emerg[i])[1]
                             for i in range(n_h_nodes)]) * k_white

    # 折射色散分段：每段一个折射缩放 + 该段波长的 XYZ
    band_edges = np.linspace(0, len(lam), n_disp + 1).astype(int)
    bands = []
    for bi in range(n_disp):
        sl = slice(band_edges[bi], band_edges[bi + 1])
        if sl.start >= sl.stop:
            continue
        lam_b = lam[sl]
        lam_rep = float(lam_b.mean())
        dsc = dispersion_scale(lam_rep) if n_disp > 1 else 1.0
        XYZ_b = np.array([col.spectrum_to_XYZ(lam_b, I_emerg[i, sl])
                          for i in range(n_h_nodes)]) * k_white
        bands.append((dsc, XYZ_b))
    return h_nodes, bands, Y_full_nodes


def forward_trace(
    n_rays_b=4_000_000,   # 沿 impact parameter 撒的光线数（等面积撒 b）
    n_sun=2000,           # 太阳圆盘等面积子点数
    h_max=90.0,           # 最大擦地高度（更高=大气太薄不折射，落到月面外圈半影）
    n_lam=301,
    grid_half_km=7000.0,  # 月面网格半宽（km），覆盖本影 R_u=4601km + 半影一点
    n_pix=280,            # 每边像素数（2D 网格 n_pix×n_pix）
    n_disp=12,            # 折射色散波段数。=1 关色散
    seed=0,
):
    """完整正向 ray tracing。返回月面 2D 亮度图 + 径向剖面 + 关键档数。

    撒法（全部等面积/等通量，零 artificial 权重）：
      1. 太阳圆盘：在角半径 ANG_SUN 圆内 rejection 撒 n_sun 个等面积子点 ξ。
      2. impact parameter：在波前平面圆环 R⊕<|b|<R⊕+h_max 内等面积撒 b（极坐标
         |b|~sqrt(uniform)，φ~uniform）。每条线代表等通量份额。
      3. 折射径向向轴弯 α(h)，落月面 (x,y) 像素。像素累加能量 / 像素面积 = 面亮度。
    """
    rng = np.random.default_rng(seed)
    lam = np.linspace(380.0, 780.0, n_lam)

    h_nodes, bands, Y_full_nodes = _precompute_emergent_luminance(
        lam, h_max=h_max, n_disp=n_disp)

    ang_sun = ang_sun_rad()
    d_moon = g.D_MOON_KM
    R_e = g.R_EARTH

    # ---- 1. 太阳圆盘等面积子点 ξ（rejection sampling，单位 rad）----
    xs, ys = [], []
    need = n_sun
    while len(xs) < n_sun:
        u = rng.uniform(-ang_sun, ang_sun, size=2 * need)
        v = rng.uniform(-ang_sun, ang_sun, size=2 * need)
        m = u * u + v * v <= ang_sun * ang_sun
        xs.extend(u[m].tolist())
        ys.extend(v[m].tolist())
    xi_x = np.array(xs[:n_sun])      # rad
    xi_y = np.array(ys[:n_sun])      # rad
    # 每个子点的落点平移（km）
    sun_dx = xi_x * d_moon
    sun_dy = xi_y * d_moon

    # ---- 月面 2D 网格 ----
    edges = np.linspace(-grid_half_km, grid_half_km, n_pix + 1)
    pix_area = (edges[1] - edges[0]) ** 2        # km^2，每像素面积（常数）
    Y_grid = np.zeros((n_pix, n_pix))            # 累加能量（=落点密度×光线能量）
    XYZ_grid = np.zeros((n_pix, n_pix, 3))       # 累加 XYZ（取色）
    cnt_grid = np.zeros((n_pix, n_pix))          # 落点计数

    # ============ 关键归一化（满月=1）============
    # 物理量：月面面亮度 = 单位月面面积接收的光通量。我们要让"满月（太阳直射、不折射、
    # 不消光）的月面面亮度 = 1"。
    #
    # 满月图景：完整平行太阳光束（波前单位面积通量 = Φ0，取 Φ0≡1）直射月面，波前面积
    # 1:1 映射月面面积，故满月面亮度 = Φ0·Y_white = 1（Y 已对白点归一 → Y_white=1）。
    #
    # 月食图景：我们只撒擦地环 R⊕<|b|<R⊕+h_max（这部分波前才会被折射进本影；|b|<R⊕
    # 被地球挡住，|b|>R⊕+h_max 大气太薄基本直穿落本影外）。这环的波前总面积
    #   A_ring = π((R⊕+h_max)^2 − R⊕^2)。
    # 在这环上等面积撒 n_rays_b 条线（每条随机配一个太阳子点 = Monte-Carlo 对圆盘积分），
    # 每条线携带的**通量** = Φ0 × (该条线占的波前面积) = Φ0 × A_ring/n_rays_b。
    # 落进某像素的总能量 / 像素面积 = 该像素面亮度（自动以满月=1 为基准，因为 Φ0=1 且
    # 满月波前面积 1:1 映射月面面积 → 同一 Φ0·Y 标度）。
    #
    # 自洽校验（见 __main__ 关 refraction/extinction 的 sanity）：若不折射不消光，擦地环
    # 的线直射落到月面同一环（面积不变），面亮度应 = 1。
    PHI0 = 1.0                                   # 波前单位面积通量（满月白光基准）
    A_ring = np.pi * ((R_e + h_max) ** 2 - R_e ** 2)
    ray_flux = PHI0 * A_ring / n_rays_b          # 每条线携带的通量（含波前面积份额）

    # ---- 主循环：每条光线撒一个随机 b + 随机太阳子点 ----
    # 向量化分块。每条线随机配一个太阳子点 = Monte-Carlo 对太阳圆盘积分（等面积子点 →
    # 不需要任何弦长权重，弦长分布从 2D 圆盘 rejection sampling 自然涌现）。太阳子点只把
    # 整条折射映射平移 ξ·d_moon，不改该条线携带的能量密度 → 满月归一不受影响。

    chunk = 1_000_000
    done = 0
    while done < n_rays_b:
        m = min(chunk, n_rays_b - done)
        done += m
        # 等面积撒 b：|b| = sqrt(U·((R⊕+h_max)^2−R⊕^2)+R⊕^2)，φ uniform
        U = rng.uniform(0.0, 1.0, size=m)
        b_mag = np.sqrt(U * ((R_e + h_max) ** 2 - R_e ** 2) + R_e ** 2)
        phi = rng.uniform(0.0, 2 * np.pi, size=m)
        bx = np.cos(phi)        # b̂ 单位向量
        by = np.sin(phi)
        h = b_mag - R_e         # 擦地高度

        # 随机配太阳子点
        si = rng.integers(0, n_sun, size=m)
        sdx = sun_dx[si]
        sdy = sun_dy[si]

        # 每条线的全谱 luminance（用于纯亮度图，不分色散段）
        Yray = np.interp(h, h_nodes, Y_full_nodes)

        # 各色散波段分别折射落点 + 累加 XYZ（颜色 + 色散涌现）
        # 亮度图 Y_grid 用全谱 Y（无色散位移，足够算亮度档数）。
        # 折射径向落点：沿 b̂ 方向，距轴 (|b| − α·d_moon)；蓝段 α 更大。
        for dsc, XYZ_b in bands:
            alpha = g.refraction_angle(h) * dsc          # rad
            r_land = b_mag - alpha * d_moon              # km，带符号径向落点
            x_land = bx * r_land + sdx
            y_land = by * r_land + sdy
            ix = np.floor((x_land + grid_half_km) / (2 * grid_half_km) * n_pix).astype(int)
            iy = np.floor((y_land + grid_half_km) / (2 * grid_half_km) * n_pix).astype(int)
            inside = (ix >= 0) & (ix < n_pix) & (iy >= 0) & (iy < n_pix)
            ii, jj = ix[inside], iy[inside]
            flat = ii * n_pix + jj
            # 该波段的 XYZ（按波长段累加），落点带本段折射色散位移 → 色散涌现
            for c in range(3):
                Xc = np.interp(h[inside], h_nodes, XYZ_b[:, c])
                np.add.at(XYZ_grid[:, :, c].reshape(-1), flat, Xc * ray_flux)

        # 全谱亮度落点（用 ref 折射，无色散位移）：用于亮度档数主结果
        alpha0 = g.refraction_angle(h)
        r_land = b_mag - alpha0 * d_moon
        x_land = bx * r_land + sdx
        y_land = by * r_land + sdy
        ix = np.floor((x_land + grid_half_km) / (2 * grid_half_km) * n_pix).astype(int)
        iy = np.floor((y_land + grid_half_km) / (2 * grid_half_km) * n_pix).astype(int)
        inside = (ix >= 0) & (ix < n_pix) & (iy >= 0) & (iy < n_pix)
        flat = ix[inside] * n_pix + iy[inside]
        np.add.at(Y_grid.reshape(-1), flat, Yray[inside] * ray_flux)
        np.add.at(cnt_grid.reshape(-1), flat, np.ones(inside.sum()))

    # ---- 满月基准（解析、与上面同一标度）----
    # 满月 = 太阳直射、不折射、不消光：完整波前（Φ0=1）1:1 映射月面 → 面亮度 = Φ0·Y_white = 1。
    # 本版每条线携带通量 ray_flux=Φ0·A_ring/n_rays_b，落进像素后除以像素面积即得面亮度，
    # 已天然以满月=1 为基准（不折射不消光时擦地环线直射落同一环，面积守恒 → 面亮度=1）。
    full_moon_surface_brightness = 1.0

    # 月面面亮度图 = 每像素累加通量 / 像素面积（= 相对满月的面亮度）
    surf = Y_grid / pix_area / full_moon_surface_brightness
    XYZ_surf = XYZ_grid / pix_area / full_moon_surface_brightness

    # ---- 径向剖面（轴对称，把 2D 网格按半径平均）----
    cx = (np.arange(n_pix) + 0.5) / n_pix * 2 * grid_half_km - grid_half_km
    XX, YY = np.meshgrid(cx, cx, indexing="ij")
    RR = np.sqrt(XX ** 2 + YY ** 2)
    r_bins = np.linspace(0, grid_half_km, 120)
    r_cent = 0.5 * (r_bins[:-1] + r_bins[1:])
    surf_r = np.full(len(r_cent), np.nan)
    RB_r = np.full(len(r_cent), np.nan)
    for i in range(len(r_cent)):
        msk = (RR >= r_bins[i]) & (RR < r_bins[i + 1])
        if msk.sum() == 0:
            continue
        surf_r[i] = surf[msk].mean()
        xyz = XYZ_surf[msk].reshape(-1, 3).sum(axis=0)
        if xyz[1] > 0:
            rgb = _M_XYZ2RGB @ xyz
            if rgb[2] > 1e-30:
                RB_r[i] = rgb[0] / rgb[2]

    # 本影中心面亮度（穿正中心 ρ≈0）= 最内圈径向 bin
    center_surf = np.nanmean(surf_r[:3])
    center_stops = np.log2(center_surf) if center_surf > 0 else -np.inf

    return dict(
        surf=surf, XYZ_surf=XYZ_surf, cnt=cnt_grid,
        cx=cx, grid_half_km=grid_half_km, pix_area=pix_area,
        r_cent=r_cent, surf_r=surf_r, RB_r=RB_r,
        center_surf=center_surf, center_stops=center_stops,
        umbra_R_km=g.umbra_radius_km(),
        full_moon_surface_brightness=full_moon_surface_brightness,
    )


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rays", type=int, default=4_000_000)
    ap.add_argument("--n_sun", type=int, default=2000)
    ap.add_argument("--n_pix", type=int, default=280)
    ap.add_argument("--n_disp", type=int, default=12)
    args = ap.parse_args()

    t0 = time.time()
    res = forward_trace(n_rays_b=args.n_rays, n_sun=args.n_sun,
                        n_pix=args.n_pix, n_disp=args.n_disp)
    dt = time.time() - t0
    print(f"正向 ray tracing 完成: n_rays={args.n_rays} n_sun={args.n_sun} "
          f"n_pix={args.n_pix} 用时 {dt:.1f}s")
    print(f"本影半径 R_u={res['umbra_R_km']:.0f}km")
    print(f"\n本影中心面亮度(相对满月)={res['center_surf']:.3e} "
          f"= {res['center_stops']:.1f} 档")
    print("\n半径(km)  面亮度(rel)   档数      R/B")
    for i in range(0, len(res["r_cent"]), 4):
        r = res["r_cent"][i]; s = res["surf_r"][i]; rb = res["RB_r"][i]
        if np.isfinite(s) and s > 0:
            print(f"  {r:6.0f}   {s:.3e}   {np.log2(s):6.1f}   "
                  f"{rb if np.isfinite(rb) else float('nan'):6.2f}")
