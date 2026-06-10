"""真·正向 ray tracer —— 月食月面绝对亮度（相对满月的档数），零 artificial 参数。

================================================================================
这是把三路真物理集成成一条完整管线的最终版本。相对旧 forward_ray_trace.py，
本版把残留的解析处方全部替换成第一性原理的数值结果：

  旧解析量（已消除）                       新版（涌现/数值积分）
  ----------------------------------------------------------------------------
  geometry.refraction_angle               refraction_trace.trace_ray —— Eikonal
    α(h)=0.0204·exp(-h/8) 解析指数律         RK4 光线积分，α 从真实 AFGL 折射率
                                            梯度涌现，含"撞地遮挡"（h<~1.72km 的
                                            光线弯到地表以下，根本到不了月面）。
  geometry.shadow_radius_signed_km          落点 = b̂·(|b| − α_traced(h)·d_moon)，
    / a_signed 解析落点映射                  α 全用真追踪值，无解析 r(h) 公式。
  geometry.column_density（直线视线）        curved_path.tau_curved —— 沿真实弯曲
    + radiative_transfer 直线 τ              光路（Bouguer 不变量）积分 τ(λ,h)。
  geometry.focusing_factor                  落点装进 2D 笛卡尔网格 ÷ 像素面积，
    = b·|dh/dr|/r + r_floor fudge            focusing 作为落点密度的真实后果涌现。
  弦长权重                                   太阳圆盘 + 波前圆环都等面积撒点，
                                            弦长/通量分布从 2D 几何自然涌现。

================================================================================
新版 focusing 怎么自然涌现
================================================================================
不再用任何解析雅可比。每条光线携带固定通量份额 ray_flux = Φ0·A_ring/n_rays，
落到月面 2D 笛卡尔网格的某像素。像素面亮度 = 累加通量 / 像素面积（常数）。

折射把擦地环 [R⊕, R⊕+h_max] 的波前非线性地映射到月面：dr_land/dh 在某些 h 处
接近 0（落点密度发散 = 焦散/聚焦），在另一些 h 处很大（落点稀疏 = 散焦）。这个
非线性 *直接* 体现为落点在网格里的疏密——聚焦区像素累加更多光线、散焦区更少。
2D 笛卡尔网格每像素面积恒定，所以不存在旧 1D 等角 bin 的"环带面积 2πρ·dρ→0"
人为塌缩，也就不需要除 2πρ、不需要 r_floor 正则化。focusing 是撒线 + 分箱的
纯粹涌现结果。

本影中心为什么暗到接近真实
================================================================================
真追踪揭示一个解析模型看不到的几何事实：α(h) 不是单调的。impact-parameter
h≲1.72km 的光线被折射弯到地表以下 → 撞地被遮挡（blocked），根本到不了月面。
真正"擦地"的最深光线是 h≈1.72km，偏转 α≈63.5'（不是解析的 70'）。本影中心
（反日轴 r=0）只能由 α·d_moon=|b| 的光线照亮，即 α≈57'、对应 h≈2.7km 的深擦边
光——它穿厚大气，单程消光在红端就 ~4 mag、蓝端 >12 mag，且这些深擦边光线在
near-axis 被强烈散焦/被撞地截断，落点密度远低于满月直射。三者叠加把中心压暗。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import geometry as g                 # 仅用常数 (R_EARTH, D_MOON, 太阳尺寸) 与 umbra_radius
import refraction_trace as rtr       # 真折射 RK4 积分 α(h) + 撞地遮挡
import curved_path as cp             # 真弯曲路径 τ(λ,h)
import solar
import color as col
import cross_sections as cs

R_EARTH = g.R_EARTH
D_MOON = g.D_MOON_KM

_M_XYZ2RGB = col.M_XYZ2RGB_LINEAR   # sRGB(D65) 线性矩阵, 单一来源(color模块)
ang_sun_rad = g.ang_sun_rad         # 太阳角半径(rad), 单一来源(geometry模块)


def dispersion_scale(lam_nm, lam_ref=600.0):
    """折射角随波长缩放 α(λ)/α(ref)。蓝端 (n-1) 大 → 折射更强 → 落点更靠外（棱镜效应）。

    注意：这不是 artificial 参数——它是干空气色散(cross_sections.dry_air_n_minus_1,
    Peck&Reeder, 与瑞利截面同一来源)的直接比值, 物理上折射率本就随波长变。
    Limitation: 假设 α(λ) 对 (n-1) 线性、路径形状消色差(~1e-4 量级可忽略); α 绝对值仍来自
    refraction_trace 的 RK4 真追踪(用600nm折射率), 这里只把它按色散比值缩放到各波段。
    """
    return cs.dry_air_n_minus_1(lam_nm) / cs.dry_air_n_minus_1(lam_ref)


def _precompute_alpha_traced(h_nodes, ds_km=0.02, z_top_km=120.0):
    """在 h 网格上预计算真追踪折射角 α(h) (rad)、真实切点高度与撞地遮挡掩码。

    返回:
      alpha_nodes : (Hn,)  真追踪 α，单位 rad；blocked 的 h 处填 nan
      z_tan_nodes : (Hn,)  真实切点高度 km。折射把切点压得比 impact-parameter h 低
                    （impact h=2.7km 的光线真切点 ≈1.1km），消光积分必须用它而非 h
      blocked     : (Hn,)  bool，True=该 impact-parameter 光线弯到地表以下被遮挡
    """
    # 矢量化批量追踪(替原串行 for, ~80× 提速, α 逐位一致)。blocked 光线 α 填 nan。
    alpha, z_tan, blocked = rtr.trace_rays_batch(h_nodes, z_top_km=z_top_km, ds_km=ds_km)
    alpha = np.where(blocked, np.nan, alpha)
    return alpha, z_tan, blocked


def _precompute_emergent_curved(h_nodes, lam, z_top_km=90.0, n_steps=4000,
                                z_tan_nodes=None, blocked_nodes=None,
                                aod550_trop=0.0, aod550_strat=0.0):
    """在 h 网格上预计算沿真实弯曲路径的出射谱 I(λ,h)=I_sun·exp(-τ_curved)。

    τ 用 curved_path.tau_curved（Bouguer 弯曲光路），替换直线柱密度。
    tau_curved 的输入语义是**切点高度**，不是 impact parameter——折射把切点压得
    比 impact-h 低，深擦边光线用 impact-h 积分会把消光低估 ~0.6-0.7 mag（本影中心
    偏亮 ~0.7 档）。调用方应传入 trace_rays_batch 给出的真实 z_tan_nodes。
    blocked 光线到不了月面，出射谱保持 0、跳过积分。
    返回 (Hn, L) 出射谱矩阵（按 impact-h 网格索引）。
    """
    I_sun = solar.solar_spectrum(lam)
    Hn = len(h_nodes)
    I_emerg = np.zeros((Hn, len(lam)))
    h_tan = h_nodes if z_tan_nodes is None else z_tan_nodes
    for i in range(Hn):
        if blocked_nodes is not None and blocked_nodes[i]:
            continue
        tau, _, _ = cp.tau_curved(float(h_tan[i]), lam, z_top_km=z_top_km,
                                  n_steps=n_steps, with_refraction=True,
                                  aod550_trop=aod550_trop,
                                  aod550_strat=aod550_strat)
        I_emerg[i] = I_sun * np.exp(-tau)
    return I_emerg


def forward_trace(
    n_rays_b=4_000_000,
    n_sun=2000,
    h_max=90.0,
    n_lam=121,
    n_h_nodes=400,          # h 网格节点数（每节点一次 RK4 追踪 + 一次弯曲 τ 积分）
    grid_half_km=7000.0,
    n_pix=280,
    n_disp=12,
    trace_ds_km=0.25,       # RK4 弧长步长（α 在 ds=0.25 vs 0.02 一致到 0.01'，12× 提速）
    tau_steps=2000,         # 弯曲路径 τ 积分步数（curved_path 在 ≥1500 步收敛）
    h_direct_max=5500.0,    # 直射光: impact b∈[R_e+h_max, R_e+h_direct_max]的光线在大气外, α=0、
                            # 满谱不衰减(太阳探出地球缘的直射光)。覆盖半影区到满月。
                            # 必须 ≥ d·tan(a_max+16')−R_e, 否则月面外缘有太阳子点的直射光
                            # 没被撒出来→亮度假性下降(采样截断, 非物理)。a=73' 需 ≥3600,
                            # 取 5500 留余量(73.2' 处实测精确回到满月 1.000)。
    refr_frac=0.5,          # 分层抽样: 折射壳层(h<h_max)的光线配额。该壳层面积只占撒线
                            # 总面积 ~1%, 均匀撒线时深本影每径向环只摊到 O(10²) 条光线,
                            # LUT/光度曲线带 MC 噪声(月盘上呈同心环假象)。每层通量按各自
                            # 面积归一(无偏 stratified sampling), 只缩方差不改物理。
    aod550_trop=0.07,       # 背景气溶胶(对流层, H=1.5km)。0=纯分子大气理论上限(比任何真实
    aod550_strat=0.005,     # 月食都亮 ~2-3 mag); 0.07+0.005=最晴夜背景(GloSSAC/AERONET 静默期,
                            # 与 Mallama 恒星测光经验消光隐含值一致)。见 2026-06-09 暗端对账。
    limb_dark=True,         # 太阳 limb darkening(V 带二次律, Allen)。盘边缘 I≈0.30·I(0),
                            # 权重归一均值=1 总通量守恒。False=均匀盘(旧行为, ablation 用)。
    seed=0,
    verbose=True,
):
    """完整真·正向 ray tracing。返回月面 2D 亮度图 + 径向剖面 + 关键档数。

    全部撒法等面积/等通量（零 artificial 权重）；α 与 τ 全用数值积分；focusing 涌现。
    """
    rng = np.random.default_rng(seed)
    lam = np.linspace(380.0, 780.0, n_lam)

    # ---- h 网格：从擦地极限以下一点到 h_max。低于擦地极限的节点会被标 blocked ----
    h_nodes = np.linspace(0.0, h_max, n_h_nodes)

    if verbose:
        print(f"[1/3] 预计算真追踪折射角 α(h)（{n_h_nodes} 个 RK4 积分）...")
    alpha_nodes, z_tan_nodes, blocked_nodes = _precompute_alpha_traced(
        h_nodes, ds_km=trace_ds_km, z_top_km=120.0)
    n_block = int(blocked_nodes.sum())
    h_graze = h_nodes[~blocked_nodes][0] if (~blocked_nodes).any() else np.nan
    if verbose:
        print(f"      撞地遮挡: {n_block}/{n_h_nodes} 个低 h 节点被遮挡 "
              f"(擦地极限 h≈{h_graze:.2f}km, α_max={np.degrees(np.nanmax(alpha_nodes))*60:.1f}')")

    if verbose:
        print(f"[2/3] 预计算弯曲路径出射谱 I(λ,h)（{n_h_nodes} 次 τ 积分）...")
    I_emerg = _precompute_emergent_curved(h_nodes, lam, z_top_km=90.0,
                                          n_steps=tau_steps,
                                          z_tan_nodes=z_tan_nodes,
                                          blocked_nodes=blocked_nodes,
                                          aod550_trop=aod550_trop,
                                          aod550_strat=aod550_strat)

    # ---- 满月白点：未衰减日光 XYZ，k_white=1/Y_white → 满月 Y=1 ----
    I_sun_full = solar.solar_spectrum(lam)
    white_XYZ = col.spectrum_to_XYZ(lam, I_sun_full)
    k_white = 1.0 / white_XYZ[1]

    # 全谱 luminance（每个 h 节点），blocked 节点 → 0（到不了月面）
    Y_full_nodes = np.array([col.spectrum_to_XYZ(lam, I_emerg[i])[1]
                             for i in range(n_h_nodes)]) * k_white
    Y_full_nodes[blocked_nodes] = 0.0

    # 折射色散分段：每段一个折射缩放 + 该段波长的 XYZ
    band_edges = np.linspace(0, len(lam), n_disp + 1).astype(int)
    bands = []
    for bi in range(n_disp):
        sl = slice(band_edges[bi], band_edges[bi + 1])
        if sl.start >= sl.stop:
            continue
        lam_b = lam[sl]
        dsc = dispersion_scale(float(lam_b.mean())) if n_disp > 1 else 1.0
        XYZ_b = np.array([col.spectrum_to_XYZ(lam_b, I_emerg[i, sl])
                          for i in range(n_h_nodes)]) * k_white
        XYZ_b[blocked_nodes] = 0.0
        bands.append((dsc, XYZ_b))

    if verbose:
        print(f"[3/3] 撒 {n_rays_b:,} 条光线 × {n_sun} 太阳子点 → 落点分箱...")

    ang_sun = ang_sun_rad()
    R_e = R_EARTH

    # ---- 太阳圆盘等面积子点 ξ（rejection sampling, rad）----
    xs, ys = [], []
    while len(xs) < n_sun:
        u = rng.uniform(-ang_sun, ang_sun, size=2 * n_sun)
        v = rng.uniform(-ang_sun, ang_sun, size=2 * n_sun)
        m = u * u + v * v <= ang_sun * ang_sun
        xs.extend(u[m].tolist())
        ys.extend(v[m].tolist())
    xi_x = np.array(xs[:n_sun]); xi_y = np.array(ys[:n_sun])
    sun_dx = xi_x * D_MOON; sun_dy = xi_y * D_MOON

    # ---- 太阳 limb darkening 权重 ----
    # V 带二次律 I(μ)/I(0)=1−0.93(1−μ)+0.23(1−μ)² (Allen, ~550nm), 盘边缘 ≈0.30。
    # 经验归一(均值=1)→总通量精确守恒, 只改盘内分布。消色差近似(色差是二阶)。
    if limb_dark:
        mu = np.sqrt(np.clip(1.0 - (xi_x ** 2 + xi_y ** 2) / ang_sun ** 2, 0.0, 1.0))
        w_ld = 1.0 - 0.93 * (1.0 - mu) + 0.23 * (1.0 - mu) ** 2
        w_ld = w_ld / w_ld.mean()
    else:
        w_ld = np.ones(n_sun)

    # ---- 月面 2D 网格 ----
    edges = np.linspace(-grid_half_km, grid_half_km, n_pix + 1)
    pix_area = (edges[1] - edges[0]) ** 2
    Y_grid = np.zeros((n_pix, n_pix))
    XYZ_grid = np.zeros((n_pix, n_pix, 3))
    cnt_grid = np.zeros((n_pix, n_pix))

    # ---- 满月归一 + 分层抽样 ----
    # 撒线范围含直射光: b∈[R_e, R_e+h_direct_max]。b<R_e+h_max 擦大气(折射), b>R_e+h_max
    # 在大气外(直射, α=0, 满谱不衰减)。半影区直射光渐入自然涌现, 平滑过渡到满月。
    # 分层(stratified): 折射壳层面积仅 ~1%, 均匀撒线会让深本影统计饥饿(同心环噪声)。
    # 两层各自等面积均匀撒、各按 flux_i=Φ0·A_i/n_i 归一 → 估计量无偏, 方差缩 ~refr_frac/面积比。
    PHI0 = 1.0
    b_outer = R_e + h_direct_max
    b_atm = R_e + h_max
    full_moon_surface_brightness = 1.0
    # 直射光的满谱 XYZ(不衰减白光, 各色散段一致——直射无色散位移)
    white_XYZ_direct = white_XYZ * k_white   # Y=1 的满月白
    n_refr = int(round(n_rays_b * refr_frac))
    strata = [(R_e, b_atm, n_refr),               # 折射壳层
              (b_atm, b_outer, n_rays_b - n_refr)]  # 直射壳层

    # B6 修复: blocked 用逐节点最近邻插值(不用单一标量阈值, 避免 blocked 区不连续时出错);
    # α 插值前把 blocked 节点用最近的 unblocked 边界 α 填充(不填 0, 避免紧邻 blocked 的
    # unblocked 光线被 0 线性混合而 α 被低估)。
    alpha_clean = alpha_nodes.copy()
    if np.isnan(alpha_clean).any() and (~blocked_nodes).any():
        first_ok = np.argmax(~blocked_nodes)          # 第一个 unblocked 节点(擦地边界)
        alpha_clean[:first_ok] = alpha_clean[first_ok]  # blocked 段用边界 α 外推填充
    alpha_clean = np.nan_to_num(alpha_clean, nan=float(alpha_clean[~blocked_nodes][-1])
                                if (~blocked_nodes).any() else 0.0)
    blocked_f = blocked_nodes.astype(float)           # 逐节点 blocked 状态(0/1)

    chunk = 1_000_000
    for b_lo, b_hi, n_stratum in strata:
        if n_stratum <= 0:
            continue
        ray_flux = PHI0 * np.pi * (b_hi ** 2 - b_lo ** 2) / n_stratum
        done = 0
        while done < n_stratum:
            m = min(chunk, n_stratum - done)
            done += m
            U = rng.uniform(0.0, 1.0, size=m)
            b_mag = np.sqrt(U * (b_hi ** 2 - b_lo ** 2) + b_lo ** 2)  # 层内等面积均匀
            phi = rng.uniform(0.0, 2 * np.pi, size=m)
            bx = np.cos(phi); by = np.sin(phi)
            h = b_mag - R_e
            direct = h > h_max          # 大气外: 直射光(α=0, 满谱不衰减)

            # 折射光: 真追踪 α(h); blocked(h<擦地极限)→撞地不落月面。直射光 α=0、永不 blocked。
            alpha0 = np.where(direct, 0.0, np.interp(h, h_nodes, alpha_clean))
            unblocked = direct | (np.interp(h, h_nodes, blocked_f) < 0.5)

            si = rng.integers(0, n_sun, size=m)
            sdx = sun_dx[si]; sdy = sun_dy[si]
            wl = w_ld[si]                      # 该光线所属太阳子点的 limb darkening 权重

            # 亮度: 折射光查 Y_full_nodes(消光后); 直射光=满月白 Y=1(不衰减)
            Yray = np.where(direct, white_XYZ_direct[1], np.interp(h, h_nodes, Y_full_nodes))

            # ---- 各色散波段：折射落点 + 累加 XYZ（颜色 + 色散涌现）----
            # 直射光 α=0(不弯不色散), 各段平摊满谱白; 折射光走色散段。
            npix2 = n_pix * n_pix
            for bi, (dsc, XYZ_b) in enumerate(bands):
                alpha = alpha0 * dsc                  # 直射 alpha0=0 → α=0
                r_land = b_mag - alpha * D_MOON
                x_land = bx * r_land + sdx
                y_land = by * r_land + sdy
                ix = np.floor((x_land + grid_half_km) / (2 * grid_half_km) * n_pix).astype(int)
                iy = np.floor((y_land + grid_half_km) / (2 * grid_half_km) * n_pix).astype(int)
                inside = unblocked & (ix >= 0) & (ix < n_pix) & (iy >= 0) & (iy < n_pix)
                flat = ix[inside] * n_pix + iy[inside]
                hh = h[inside]; dir_in = direct[inside]; wl_in = wl[inside]
                for c in range(3):
                    # 折射光: 查该段消光后 XYZ; 直射光: 满谱白的该段份额(满谱/n_disp, 平摊到各段)
                    Xc = np.interp(hh, h_nodes, XYZ_b[:, c])
                    Xc = np.where(dir_in, white_XYZ_direct[c] / len(bands), Xc)
                    XYZ_grid[:, :, c].reshape(-1)[:] += np.bincount(
                        flat, weights=Xc * wl_in * ray_flux, minlength=npix2)

            # ---- 全谱亮度落点（用 ref 折射，无色散位移）：亮度档数主结果 ----
            r_land = b_mag - alpha0 * D_MOON
            x_land = bx * r_land + sdx
            y_land = by * r_land + sdy
            ix = np.floor((x_land + grid_half_km) / (2 * grid_half_km) * n_pix).astype(int)
            iy = np.floor((y_land + grid_half_km) / (2 * grid_half_km) * n_pix).astype(int)
            inside = unblocked & (ix >= 0) & (ix < n_pix) & (iy >= 0) & (iy < n_pix)
            flat = ix[inside] * n_pix + iy[inside]
            npix2 = n_pix * n_pix
            Y_grid.reshape(-1)[:] += np.bincount(
                flat, weights=(Yray * wl)[inside] * ray_flux, minlength=npix2)
            cnt_grid.reshape(-1)[:] += np.bincount(flat, minlength=npix2)

    # ---- 面亮度 = 每像素累加通量 / 像素面积（相对满月）----
    surf = Y_grid / pix_area / full_moon_surface_brightness
    XYZ_surf = XYZ_grid / pix_area / full_moon_surface_brightness

    # ---- 径向剖面（轴对称，按半径平均）----
    cx = (np.arange(n_pix) + 0.5) / n_pix * 2 * grid_half_km - grid_half_km
    XX, YY = np.meshgrid(cx, cx, indexing="ij")
    RR = np.sqrt(XX ** 2 + YY ** 2)
    r_bins = np.linspace(0, grid_half_km, 120)
    r_cent = 0.5 * (r_bins[:-1] + r_bins[1:])
    surf_r = np.full(len(r_cent), np.nan)
    RB_r = np.full(len(r_cent), np.nan)
    XYZ_r = np.full((len(r_cent), 3), np.nan)   # 各半径的平均 XYZ(供建 a→XYZ LUT)
    for i in range(len(r_cent)):
        msk = (RR >= r_bins[i]) & (RR < r_bins[i + 1])
        if msk.sum() == 0:
            continue
        surf_r[i] = surf[msk].mean()
        XYZ_r[i] = XYZ_surf[msk].reshape(-1, 3).mean(axis=0)
        xyz = XYZ_r[i]
        if xyz[1] > 0:
            rgb = _M_XYZ2RGB @ xyz
            if rgb[2] > 1e-30:
                RB_r[i] = rgb[0] / rgb[2]

    center_surf = np.nanmean(surf_r[:3])
    center_stops = np.log2(center_surf) if center_surf > 0 else -np.inf

    return dict(
        surf=surf, XYZ_surf=XYZ_surf, cnt=cnt_grid,
        cx=cx, grid_half_km=grid_half_km, pix_area=pix_area,
        r_cent=r_cent, surf_r=surf_r, RB_r=RB_r, XYZ_r=XYZ_r,
        center_surf=center_surf, center_stops=center_stops,
        umbra_R_km=g.umbra_radius_km(),
        h_graze_km=h_graze, n_blocked_nodes=n_block,
        alpha_nodes=alpha_nodes, h_nodes=h_nodes, blocked_nodes=blocked_nodes,
        full_moon_surface_brightness=full_moon_surface_brightness,
    )


def build_lut_from_raytrace(res=None, a_hi=73.0, smooth_sigma_bins=2.0, **trace_kw):
    """从真 ray tracing 径向剖面建 a(arcmin)→XYZ LUT, 兼容 render_rt.shade_disk_lut 接口。

    res: forward_trace 结果(None 则现跑一次)。返回 dict(a, XYZ): a 角距(arcmin), XYZ 线性。
    含半影直射光: 本影中心(古铜血月)→半影区直射光渐入→出本影满月, 全程ray tracing涌现,
    **无硬clamp**(直射光实现后剖面自然到满月)。替换旧 build_disk_lut(偏亮-7.7)。
    默认 grid/h_direct_max 覆盖满月(73')。
    """
    import numpy as _np
    if res is None:
        # 默认参数覆盖满月: grid_half_km 大、h_direct_max 含直射光且覆盖 a_hi+16'
        # (h_direct_max=3000 只干净到 ~68', 曾让 LUT 外缘 68-73' 满月端假性偏暗 ~0.2 档)
        trace_kw.setdefault("grid_half_km", 9000.0)
        trace_kw.setdefault("h_direct_max", 5500.0)
        res = forward_trace(verbose=False, **trace_kw)
    rc_km = _np.asarray(res["r_cent"]); XYZ_r = _np.asarray(res["XYZ_r"])
    a = _np.degrees(_np.arctan(rc_km / D_MOON)) * 60.0
    ok = _np.isfinite(XYZ_r[:, 1]) & (a <= a_hi)
    a = a[ok]; XYZ = XYZ_r[ok].copy()
    if smooth_sigma_bins and smooth_sigma_bins > 0:
        # 残余蒙特卡洛分箱噪声平滑(续15 同款处理): σ=2bins≈0.6', 远小于青带 ~4' 与
        # 悬崖 ~3.5' 的物理尺度, 不毁结构; 主治本影深处低统计 bin 经 LUT 插值放大成
        # 月盘同心环的假象。分层抽样(refr_frac)已消大半噪声, 这是第二道保险。0=关。
        from scipy.ndimage import gaussian_filter1d
        for c in range(3):
            XYZ[:, c] = gaussian_filter1d(XYZ[:, c], sigma=smooth_sigma_bins,
                                          mode="nearest")
    return dict(a=a, XYZ=XYZ)


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_rays", type=int, default=4_000_000)
    ap.add_argument("--n_sun", type=int, default=2000)
    ap.add_argument("--n_pix", type=int, default=280)
    ap.add_argument("--n_disp", type=int, default=12)
    ap.add_argument("--n_h", type=int, default=600)
    ap.add_argument("--tau_steps", type=int, default=4000)
    args = ap.parse_args()

    t0 = time.time()
    res = forward_trace(n_rays_b=args.n_rays, n_sun=args.n_sun, n_pix=args.n_pix,
                        n_disp=args.n_disp, n_h_nodes=args.n_h, tau_steps=args.tau_steps)
    dt = time.time() - t0
    arcmin = lambda a: np.degrees(a) * 60.0
    print(f"\n真·正向 ray tracing 完成: n_rays={args.n_rays:,} n_sun={args.n_sun} "
          f"n_pix={args.n_pix} 用时 {dt:.1f}s")
    print(f"本影半径 R_u={res['umbra_R_km']:.0f}km")
    print(f"擦地极限 h≈{res['h_graze_km']:.2f}km, 被遮挡低 h 节点 {res['n_blocked_nodes']}")
    print(f"\n本影中心面亮度(相对满月)={res['center_surf']:.3e} = {res['center_stops']:.2f} 档")
    print("对照真实穿正中心 -14~-19 档 (arXiv 2112.08966)")
    print("\n半径(km)  面亮度(rel)   档数      R/B")
    for i in range(0, len(res["r_cent"]), 3):
        r = res["r_cent"][i]; s = res["surf_r"][i]; rb = res["RB_r"][i]
        if np.isfinite(s) and s > 0:
            print(f"  {r:6.0f}   {s:.3e}   {np.log2(s):6.2f}   "
                  f"{rb if np.isfinite(rb) else float('nan'):7.3f}")
