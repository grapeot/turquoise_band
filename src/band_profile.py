"""波段径向剖面 + R/B 口径统一 + 模型卡 —— 与文献可对账的量, 单一出处。

由暗端对账临时脚本(tmp/turquoise_critique/darkend/scripts/band_profiles.py)收编而来,
物理与 raytrace_eclipse.forward_trace 完全同源(同撒线机制、同插值、同 seed 可复现),
把每条折射光线的权重从 XYZ 换成波段带内积分辐射, 用于:

  1. Johnson-Cousins 近似带(B/V/R/I) magnitudes-lost 径向剖面(对账 Mallama 2022,
     arXiv:2112.08966) + photopic 本影中心档数;
  2. Shu 2024 / GF-4 窄带(B2=450-520nm, B4=630-690nm, 中心 491/654) R/B 径向剖面,
     三种口径显式分解(见 rb_shu_profile docstring)。实证注意: 单层窄带凹陷(654/491
     透射比 ~0.87)经 32' 太阳盘卷积+本影红层流入后在径向平均上不存活, 文献 ribbon
     当前复现不出, 分析见 docs/MODEL_CARD.md "Shu 口径裁决";
  3. ext/geo 双网格: ext=带消光权重, geo=同几何但 T≡1(关消光), ext/geo=有效透射
     —— 稀释(几何)与消光两因子的分解能力;
  4. `--model-card` 一键产出 docs/MODEL_CARD.md(当前模型关键数字 + 口径标注)。

与 forward_trace 的物理对齐(默认全开, 可关):
  - 气溶胶: precompute 阶段对每个 unblocked 节点沿弯曲路径积分 550nm 气溶胶
    光学厚度两组分 tau_t550/tau_s550(curved_path.tau_aer550_components), 权重
    函数里 tau += tau_t550·(λ/550)^(−ALPHA_TROP) + tau_s550·(λ/550)^(−ALPHA_STRAT)
    (α 分层 0.7/2.0, 2026-06-10 裁决, 替代旧单一 α=1.3)。默认 aod550_trop=0.07 /
    aod550_strat=0.005, 与 forward_trace 一致。
  - 太阳 limb darkening: scatter 的太阳子点权重 w_ld = 1−0.93(1−μ)+0.23(1−μ)²
    (Allen V 带二次律), 均值归一(总通量守恒), 与 forward_trace 同公式。

数据覆盖确认: SAO2010 太阳谱原始文件到 1001nm(src/solar.py 只载 360-820, 这里自带
loader 载到 950); Serdyuchenko 臭氧原始到 1100nm(data_loaders 截到 830, >830 填 0,
I 带内臭氧本可忽略); AFGL 到 120km; 瑞利截面解析任意 λ。
注意: 管线消光只有 Rayleigh+O3+背景气溶胶, I 带内真实大气还有 H2O 720/820nm 吸收带
未建模 → I 带绝对值偏亮, 模型卡不引用。

跑:
  source .venv/bin/activate
  python src/band_profile.py --model-card           # 默认参数(气溶胶+LD 开), 写 docs/MODEL_CARD.md
  python src/band_profile.py --n-rays 1000000       # 快速剖面表(不写模型卡)
"""
import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import geometry as g
import refraction_trace as rtr
import curved_path as cp
import cross_sections as cs
import color as col
import atmosphere as atm

_ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(_ROOT, "data", "raw")
DOCS = os.path.join(_ROOT, "docs")
OUTPUTS = os.path.join(_ROOT, "outputs")

R_EARTH = g.R_EARTH
D_MOON = g.D_MOON_KM
R_UMBRA = g.umbra_radius_km()
KM_PER_ARCMIN = 112.0      # 月面尺度(与 measure_ribbon 一致)

# 带定义(nm): BVRI=Johnson-Cousins 近似(对账 Mallama); B2/B4=Shu 2024 / GF-4 窄带
BANDS = {"B": (390.0, 490.0), "V": (500.0, 600.0),
         "R": (570.0, 720.0), "I": (720.0, 880.0),
         "B2": (450.0, 520.0), "B4": (630.0, 690.0)}

# 气溶胶 Ångström 指数: 分层 atm.ALPHA_TROP=0.7 / atm.ALPHA_STRAT=2.0
# (单一 ANGSTROM_EXP=1.3 已于 2026-06-10 退役, 见 atmosphere.py 常数注释与 DECISION 裁决)


def load_solar_wide(lo=360.0, hi=950.0):
    """SAO2010 太阳谱宽带 loader(src/solar.py 只载到 820nm, 盖不住 I 带)。"""
    from scipy.interpolate import interp1d
    d = np.loadtxt(os.path.join(RAW, "sao2010_solref.dat"), comments="C")
    lam, irr = d[:, 0], d[:, 2]
    m = (lam >= lo) & (lam <= hi)
    return interp1d(lam[m], irr[m], bounds_error=False, fill_value=0.0)


SUN = load_solar_wide()


def sun_band_ratio(num="B4", den="B2", per_nm=True):
    """太阳谱带比值 ∫_num I dλ / ∫_den I dλ。

    per_nm=True: 带平均(每 nm)谱辐射比——卫星带辐亮度(W/m²/sr/μm)的口径,
    rb_shu_profile 用它作"太阳谱斜率"因子 S。B4/B2 带平均比 ≈0.80;
    单色中心比 654/491 ≈0.737(working.md 勘误2 引的数); 带积分(不除带宽)≈0.685
    ——三个数差到 8-15%, 口径必须写明。
    """
    lo_n, hi_n = BANDS[num]
    lo_d, hi_d = BANDS[den]
    lam_n = np.arange(lo_n, hi_n + 0.5, 1.0)
    lam_d = np.arange(lo_d, hi_d + 0.5, 1.0)
    i_n = np.trapezoid(SUN(lam_n), lam_n)
    i_d = np.trapezoid(SUN(lam_d), lam_d)
    if per_nm:
        return float((i_n / (hi_n - lo_n)) / (i_d / (hi_d - lo_d)))
    return float(i_n / i_d)


def precompute_nodes(n_h_nodes=400, h_max=90.0, trace_ds_km=0.25, tau_steps=2000,
                     aod550_trop=0.07, aod550_strat=0.005):
    """h 网格上的 α(h)、z_tan、blocked、弯曲路径柱密度 N_air/N_o3、
    气溶胶 slant 光学厚度两组分 tau_t550/tau_s550。

    完全照搬 raytrace_eclipse: α 用 trace_rays_batch(z_top=120), 消光柱密度用
    tau_curved(z_tan, z_top=90)——真实切点高度口径(2026-06-09 修复1)。
    气溶胶: curved_path.tau_aer550_components 沿同一弯曲路径几何积分两组分
    (对流层/平流层, Ångström 指数不同必须分存; 2026-06-10 裁决, 替代旧的
    单条 tau_aer550 减法提取)。aod 置 0 时两组分恒为 0。
    """
    h_nodes = np.linspace(0.0, h_max, n_h_nodes)
    alpha, z_tan, blocked = rtr.trace_rays_batch(h_nodes, z_top_km=120.0,
                                                 ds_km=trace_ds_km)
    alpha = np.where(blocked, np.nan, alpha)
    N_air = np.zeros(n_h_nodes)
    N_o3 = np.zeros(n_h_nodes)
    tau_t550 = np.zeros(n_h_nodes)
    tau_s550 = np.zeros(n_h_nodes)
    lam550 = np.array([550.0])
    for i in range(n_h_nodes):
        if blocked[i]:
            continue
        _, Na, No = cp.tau_curved(float(z_tan[i]), lam550, z_top_km=90.0,
                                  n_steps=tau_steps, with_refraction=True)
        N_air[i] = Na
        N_o3[i] = No
        if aod550_trop > 0.0 or aod550_strat > 0.0:
            tau_t550[i], tau_s550[i] = cp.tau_aer550_components(
                float(z_tan[i]), z_top_km=90.0, n_steps=tau_steps,
                with_refraction=True, aod550_trop=aod550_trop,
                aod550_strat=aod550_strat)
    return dict(h_nodes=h_nodes, alpha=alpha, z_tan=z_tan, blocked=blocked,
                N_air=N_air, N_o3=N_o3, tau_t550=tau_t550, tau_s550=tau_s550,
                aod550_trop=aod550_trop, aod550_strat=aod550_strat)


def _tau_nodes(nodes, lam):
    """节点 × 波长的总光学厚度矩阵: Rayleigh + O3 + 气溶胶两组分(独立 α)。"""
    lam = np.asarray(lam, float)
    return (nodes["N_air"][:, None] * cs.sigma_rayleigh(lam)[None, :]
            + nodes["N_o3"][:, None] * cs.sigma_o3(lam)[None, :]
            + nodes["tau_t550"][:, None] * (lam[None, :] / 550.0) ** (-atm.ALPHA_TROP)
            + nodes["tau_s550"][:, None] * (lam[None, :] / 550.0) ** (-atm.ALPHA_STRAT))


def band_weights(nodes, bands=None):
    """每带: w(h) = ∫_band I_sun·e^−τ dλ / ∫_band I_sun dλ(满月=1, 带内归一),
    dsc = 该带通量加权平均 λ 的色散缩放。τ 含 Rayleigh+O3+气溶胶。blocked 节点 w=0。

    注意 w 是带内归一的有效透射(每带各自以满月为 1), 跨带的太阳谱斜率已除掉
    —— 跨带辐亮度比要把 sun_band_ratio 乘回去, 见 rb_shu_profile。
    """
    if bands is None:
        bands = BANDS
    out = {}
    for name, (lo, hi) in bands.items():
        lam_b = np.arange(lo, hi + 0.5, 1.0)
        I0 = SUN(lam_b)
        denom = np.trapezoid(I0, lam_b)
        tau = _tau_nodes(nodes, lam_b)
        w = np.trapezoid(I0[None, :] * np.exp(-tau), lam_b, axis=1) / denom
        w[nodes["blocked"]] = 0.0
        lam_mean = float(np.trapezoid(lam_b * I0, lam_b) / denom)
        dsc = float(cs.dry_air_n_minus_1(lam_mean) / cs.dry_air_n_minus_1(600.0))
        out[name] = dict(w=w, dsc=dsc, lam_mean=lam_mean)
    return out


def mono_weights(nodes, lam0_nm):
    """单波长透射权重 w(h)=e^−τ(λ0)(满月=1)与色散缩放。

    GF-4 有效波长口径: 卫星带辐亮度对平滑谱 ≈ 有效波长处的谱辐亮度
    (B2→491nm, B4→654nm)。与 boxcar 带积分(band_weights)是两种 band 建模,
    Chappuis 对比深度差别可观(单色 654/491 透射比最深 ~0.87 vs boxcar ~0.93),
    模型卡两种都报。
    """
    lam = np.array([float(lam0_nm)])
    tau = _tau_nodes(nodes, lam)
    w = np.exp(-tau[:, 0])
    w[nodes["blocked"]] = 0.0
    dsc = float(cs.dry_air_n_minus_1(float(lam0_nm)) / cs.dry_air_n_minus_1(600.0))
    return dict(w=w, dsc=dsc, lam_mean=float(lam0_nm))


def photopic_weights(nodes, n_lam=121):
    """photopic Y 权重(对账 raytrace_eclipse 本影中心档数口径)。τ 含气溶胶。"""
    lam = np.linspace(380.0, 780.0, n_lam)
    I0 = SUN(lam)
    k_white = 1.0 / col.spectrum_to_XYZ(lam, I0)[1]
    tau = _tau_nodes(nodes, lam)
    w = np.zeros(len(nodes["h_nodes"]))
    for i in range(len(w)):
        if nodes["blocked"][i]:
            continue
        w[i] = col.spectrum_to_XYZ(lam, I0 * np.exp(-tau[i]))[1] * k_white
    return w


def scatter(nodes, channels, n_rays_b=4_000_000, n_sun=2000, h_max=90.0,
            h_direct_max=5500.0, grid_half_km=9000.0, n_pix=360, n_r_bins=160,
            limb_dark=True, seed=0, center_stats_band=None, center_r_km=300.0):
    """撒线(机制照搬 forward_trace)。channels: list of dict(name, w, dsc)。

    每个 channel 同时累计两套网格: ext(权重 w)与 geo(权重 1, 同几何)——
    geo 即"关消光"(T≡1), ext/geo = 有效透射(稀释/消光分解)。
    limb_dark: 太阳子点 limb darkening 权重(同 forward_trace 公式, 均值归一),
    源分布属性, ext 与 geo 同乘(不影响二者比值的语义)。
    center_stats_band: 用该 channel 的几何收集 r<center_r_km 折射光线 impact-h
    直方图(unweighted ray 计数 + 该带通量加权)。
    """
    rng = np.random.default_rng(seed)
    h_nodes = nodes["h_nodes"]
    blocked_f = nodes["blocked"].astype(float)
    # blocked 段 α 用最近 unblocked 边界值填充(同 forward_trace B6 修复)
    alpha_clean = nodes["alpha"].copy()
    nb = nodes["blocked"]
    if np.isnan(alpha_clean).any() and (~nb).any():
        first_ok = np.argmax(~nb)
        alpha_clean[:first_ok] = alpha_clean[first_ok]
    alpha_clean = np.nan_to_num(
        alpha_clean, nan=float(alpha_clean[~nb][-1]) if (~nb).any() else 0.0)

    # ---- 太阳圆盘等面积子点 + limb darkening 权重 ----
    ang_sun = g.ang_sun_rad()
    if n_sun == 1:
        xi_x = np.zeros(1)
        xi_y = np.zeros(1)     # 点源 = 太阳圆盘中心
    else:
        xs, ys = [], []
        while len(xs) < n_sun:
            u = rng.uniform(-ang_sun, ang_sun, size=2 * n_sun)
            v = rng.uniform(-ang_sun, ang_sun, size=2 * n_sun)
            m = u * u + v * v <= ang_sun * ang_sun
            xs.extend(u[m].tolist())
            ys.extend(v[m].tolist())
        xi_x = np.array(xs[:n_sun])
        xi_y = np.array(ys[:n_sun])
    sun_dx = xi_x * D_MOON
    sun_dy = xi_y * D_MOON
    if limb_dark:
        mu = np.sqrt(np.clip(1.0 - (xi_x ** 2 + xi_y ** 2) / ang_sun ** 2, 0.0, 1.0))
        w_ld = 1.0 - 0.93 * (1.0 - mu) + 0.23 * (1.0 - mu) ** 2
        w_ld = w_ld / w_ld.mean()
    else:
        w_ld = np.ones(len(xi_x))

    edges = np.linspace(-grid_half_km, grid_half_km, n_pix + 1)
    pix_area = (edges[1] - edges[0]) ** 2
    b_outer = R_EARTH + h_direct_max
    A_ring = np.pi * (b_outer ** 2 - R_EARTH ** 2)
    ray_flux = A_ring / n_rays_b
    npix2 = n_pix * n_pix

    grids = {c["name"]: dict(ext=np.zeros(npix2), geo=np.zeros(npix2))
             for c in channels}
    h_hist_edges = np.arange(0.0, 30.0001, 0.05)
    h_hist_cnt = np.zeros(len(h_hist_edges) - 1)
    h_hist_flux = np.zeros(len(h_hist_edges) - 1)

    chunk = 1_000_000
    done = 0
    while done < n_rays_b:
        m = min(chunk, n_rays_b - done)
        done += m
        U = rng.uniform(0.0, 1.0, size=m)
        b_mag = np.sqrt(U * (b_outer ** 2 - R_EARTH ** 2) + R_EARTH ** 2)
        phi = rng.uniform(0.0, 2 * np.pi, size=m)
        bx = np.cos(phi); by = np.sin(phi)
        h = b_mag - R_EARTH
        direct = h > h_max
        alpha0 = np.where(direct, 0.0, np.interp(h, h_nodes, alpha_clean))
        unblocked = direct | (np.interp(h, h_nodes, blocked_f) < 0.5)
        si = rng.integers(0, len(sun_dx), size=m)
        sdx = sun_dx[si]; sdy = sun_dy[si]
        wl = w_ld[si]

        for c in channels:
            alpha = alpha0 * c["dsc"]
            r_land = b_mag - alpha * D_MOON
            x_land = bx * r_land + sdx
            y_land = by * r_land + sdy
            ix = np.floor((x_land + grid_half_km) / (2 * grid_half_km) * n_pix).astype(int)
            iy = np.floor((y_land + grid_half_km) / (2 * grid_half_km) * n_pix).astype(int)
            inside = unblocked & (ix >= 0) & (ix < n_pix) & (iy >= 0) & (iy < n_pix)
            flat = ix[inside] * n_pix + iy[inside]
            hh = h[inside]; dir_in = direct[inside]; wl_in = wl[inside]
            w = np.where(dir_in, 1.0, np.interp(hh, h_nodes, c["w"]))
            grids[c["name"]]["ext"] += np.bincount(flat, weights=w * wl_in * ray_flux,
                                                   minlength=npix2)
            grids[c["name"]]["geo"] += np.bincount(flat, weights=wl_in * ray_flux,
                                                   minlength=npix2)
            if center_stats_band == c["name"]:
                r_tot = np.sqrt(x_land[inside] ** 2 + y_land[inside] ** 2)
                cm = (r_tot < center_r_km) & ~dir_in
                h_hist_cnt += np.histogram(hh[cm], bins=h_hist_edges)[0]
                h_hist_flux += np.histogram(hh[cm], bins=h_hist_edges,
                                            weights=(w * wl_in)[cm])[0]

    # ---- 径向剖面(同 forward_trace: 2D 像素值按半径环平均) ----
    cx = (np.arange(n_pix) + 0.5) / n_pix * 2 * grid_half_km - grid_half_km
    XX, YY = np.meshgrid(cx, cx, indexing="ij")
    RR = np.sqrt(XX ** 2 + YY ** 2).reshape(-1)
    r_bins = np.linspace(0, grid_half_km, n_r_bins)
    r_cent = 0.5 * (r_bins[:-1] + r_bins[1:])
    prof = {}
    for name, gset in grids.items():
        prof[name] = {}
        for kind in ("ext", "geo"):
            surf = gset[kind] / pix_area
            pr = np.full(len(r_cent), np.nan)
            for i in range(len(r_cent)):
                msk = (RR >= r_bins[i]) & (RR < r_bins[i + 1])
                if msk.sum():
                    pr[i] = surf[msk].mean()
            prof[name][kind] = pr
        prof[name]["surf2d_ext"] = grids[name]["ext"].reshape(n_pix, n_pix) / pix_area
    return dict(r_cent=r_cent, prof=prof,
                h_hist_edges=h_hist_edges, h_hist_cnt=h_hist_cnt,
                h_hist_flux=h_hist_flux, pix_area=pix_area)


# ─────────────────────────────── R/B 口径统一 ───────────────────────────────

def rb_calibers(prof_b4_ext, prof_b2_ext, sun_ratio, albedo_ratio=1.35):
    """从 B4/B2 带内归一剖面算三种 R/B 口径。返回 dict(raw, sun_norm, shu)。

    显式分解(径向每点):
      rb_sun_norm = P_B4/P_B2          带内归一比 = 有效透射比 T_B4/T_B2。
                                       中性白=1 口径: 直射区(满月)恒等于 1。
      rb_raw      = rb_sun_norm × S    带平均辐亮度直除比(灰月面), 含太阳谱斜率。
                                       S = sun_band_ratio(per_nm)≈0.80, 直射区=S。
      rb_shu      = rb_raw × A         Shu 2024 卫星 radiance 直除口径:
                                       再乘月面反照率红坡 A=albedo_ratio(654/491,
                                       默认 1.35, 月海/高地随 Ti 含量 ±15%)。
                                       直射区 = S·A ≈ 1.08。

    与文献比较用 rb_shu(Shu 的 R/B<1 ribbon、in-band 0.8-1.0 都是这个口径);
    rb_sun_norm 是续16 的"中性白"度量; rb_raw 是模型灰月面辐亮度比, 基线 0.80<1,
    其"R/B<1"无界、不构成 ribbon。三者只差常数因子但语义不同, 绝不可混比。
    注: working.md 2026-06-09 勘误2 说"太阳斜率×反照率红坡近似相消"用的是单色中心比
    0.737×1.35≈1.0; 带平均口径 S≈0.80 → S·A≈1.08, 相消到 8% 以内(小于反照率 ±15%
    系统不确定度)。渲染管线(build_video/render_rt/render_textured)绝不做任何
    sun-normalization——那是对齐卫星定标的测量口径, 不是人眼物理。
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        rb_sun_norm = np.where(prof_b2_ext > 0, prof_b4_ext / prof_b2_ext, np.nan)
    rb_raw = rb_sun_norm * sun_ratio
    rb_shu = rb_raw * albedo_ratio
    return dict(raw=rb_raw, sun_norm=rb_sun_norm, shu=rb_shu)


def rb_shu_profile(n_rays_b=4_000_000, n_sun=2000, n_h_nodes=400, h_max=90.0,
                   trace_ds_km=0.25, tau_steps=2000, grid_half_km=9000.0,
                   n_pix=360, n_r_bins=160, aod550_trop=0.07, aod550_strat=0.005,
                   limb_dark=True, albedo_ratio=1.35, seed=0,
                   nodes=None, scatter_res=None, verbose=True):
    """Shu 2024 / GF-4 口径的 R/B 径向剖面, 三种口径显式分解。

    返回 dict:
      r_cent      : 径向 bin 中心 (km)
      a_arcmin    : 对应月面角距 (arcmin)
      rb_raw      : 带平均辐亮度直除比(灰月面, 含太阳谱斜率; 直射区=S≈0.80)
      rb_sun_norm : ÷ 太阳谱带平均比 S(中性白=1 口径; 直射区=1)
      rb_shu      : × 月面反照率比 654/491(albedo_ratio, 默认 1.35 ±15% 月海/高地)
                    = 与 Shu 卫星 radiance 直除口径对齐的量(直射区 = S·A ≈1.08)
      sun_ratio / albedo_ratio / prof(B2,B4 的 ext/geo 剖面) / surf_phot(若有)

    口径指南(详见 rb_calibers docstring): 与文献比较(Shu 的 R/B<1 ribbon 位置、
    in-band 0.8-1.0)用 rb_shu; rb_raw 基线 0.80<1, 其 "<1" 区间无界、无 ribbon 语义;
    渲染管线绝不做任何 sun-normalization。
    nodes / scatter_res 可传入复用(模型卡一次撒线多处取数)。
    """
    if nodes is None:
        if verbose:
            print(f"[rb_shu_profile] precompute_nodes(n_h={n_h_nodes}, "
                  f"aod={aod550_trop}/{aod550_strat}) ...")
        nodes = precompute_nodes(n_h_nodes, h_max, trace_ds_km, tau_steps,
                                 aod550_trop=aod550_trop, aod550_strat=aod550_strat)
    if scatter_res is None:
        bw = band_weights(nodes, {k: BANDS[k] for k in ("B2", "B4")})
        channels = [dict(name=k, w=v["w"], dsc=v["dsc"]) for k, v in bw.items()]
        # 单色 λ_eff 变体(GF-4 有效波长口径), 见 mono_weights docstring
        channels.append(dict(name="B2c", **mono_weights(nodes, 491.0)))
        channels.append(dict(name="B4c", **mono_weights(nodes, 654.0)))
        if verbose:
            print(f"[rb_shu_profile] 撒线 {n_rays_b:,} × n_sun={n_sun} "
                  f"(limb_dark={limb_dark}) ...")
        scatter_res = scatter(nodes, channels, n_rays_b=n_rays_b, n_sun=n_sun,
                              h_max=h_max, grid_half_km=grid_half_km, n_pix=n_pix,
                              n_r_bins=n_r_bins, limb_dark=limb_dark, seed=seed)
    prof = scatter_res["prof"]
    S = sun_band_ratio("B4", "B2", per_nm=True)
    rb = rb_calibers(prof["B4"]["ext"], prof["B2"]["ext"], S, albedo_ratio)
    r_cent = scatter_res["r_cent"]
    a_arcmin = np.degrees(np.arctan(r_cent / D_MOON)) * 60.0
    out = dict(r_cent=r_cent, a_arcmin=a_arcmin,
               rb_raw=rb["raw"], rb_sun_norm=rb["sun_norm"], rb_shu=rb["shu"],
               sun_ratio=S, albedo_ratio=albedo_ratio, prof=prof,
               scatter_res=scatter_res, nodes=nodes)
    if "B2c" in prof and "B4c" in prof:
        S_mono = float(SUN(654.0) / SUN(491.0))
        rbm = rb_calibers(prof["B4c"]["ext"], prof["B2c"]["ext"], S_mono,
                          albedo_ratio)
        out.update(rb_raw_mono=rbm["raw"], rb_sun_norm_mono=rbm["sun_norm"],
                   rb_shu_mono=rbm["shu"], sun_ratio_mono=S_mono)
    return out


def measure_rb_ribbon(a_arcmin, rb, surf=None, surf_floor=1e-7):
    """量 R/B<1 ribbon: 全局最蓝点 + 向内/向外与 1.0 的交点(线性插值)。

    返回 dict(a_min, rb_min, inner, outer, width_arcmin, width_km, bounded)。
    outer=nan 表示向外不再回到 >1(该口径基线 <1, ribbon 无界——raw 口径即如此)。
    surf 给亮度剖面时, 只在 surf>surf_floor 的可测区内找(剔除空 bin 噪声)。
    """
    rb = np.asarray(rb, float)
    valid = np.isfinite(rb)
    if surf is not None:
        valid &= np.asarray(surf) > surf_floor
    if not valid.any():
        return dict(a_min=np.nan, rb_min=np.nan, inner=np.nan, outer=np.nan,
                    width_arcmin=np.nan, width_km=np.nan, bounded=False)
    idx = np.where(valid)[0]
    i_min = idx[np.argmin(rb[idx])]
    if rb[i_min] >= 1.0:        # 全程不过 1: 无 ribbon, 但仍报最蓝点
        return dict(a_min=float(a_arcmin[i_min]), rb_min=float(rb[i_min]),
                    inner=np.nan, outer=np.nan, width_arcmin=np.nan,
                    width_km=np.nan, bounded=False)

    def _cross(i0, step):
        i = i0
        while 0 <= i + step < len(rb):
            j = i + step
            if not valid[j]:
                return np.nan
            if rb[j] >= 1.0:
                # 线性插值过 1.0
                f = (1.0 - rb[i]) / (rb[j] - rb[i])
                return float(a_arcmin[i] + f * (a_arcmin[j] - a_arcmin[i]))
            i = j
        return np.nan

    inner = _cross(i_min, -1)
    outer = _cross(i_min, +1)
    bounded = np.isfinite(inner) and np.isfinite(outer)
    width = (outer - inner) if bounded else np.nan
    return dict(a_min=float(a_arcmin[i_min]), rb_min=float(rb[i_min]),
                inner=inner, outer=outer, width_arcmin=width,
                width_km=width * KM_PER_ARCMIN if bounded else np.nan,
                bounded=bounded)


def cliff_width(a_arcmin, surf, lo=0.1, hi=0.9):
    """亮度悬崖宽: photopic 面亮度(相对满月)从 lo 升到 hi 的径向宽度(arcmin)。

    在外侧上升沿量: 找第一个 surf≥hi 的 bin, 向内回溯最后一个 surf<lo 的 bin,
    两交点都在 log10(surf) 上线性插值(跨多个量级更稳)。
    """
    surf = np.asarray(surf, float)
    v = np.isfinite(surf) & (surf > 0)
    i_hi = None
    for i in range(len(surf)):
        if v[i] and surf[i] >= hi:
            i_hi = i
            break
    if i_hi is None or i_hi == 0:
        return dict(a_lo=np.nan, a_hi=np.nan, width_arcmin=np.nan)

    def _interp_cross(i, j, level):
        ls = np.log10([max(surf[i], 1e-30), max(surf[j], 1e-30)])
        f = (np.log10(level) - ls[0]) / (ls[1] - ls[0])
        return float(a_arcmin[i] + f * (a_arcmin[j] - a_arcmin[i]))

    a_hi_x = _interp_cross(i_hi - 1, i_hi, hi) if v[i_hi - 1] else float(a_arcmin[i_hi])
    i_lo = None
    for i in range(i_hi, -1, -1):
        if v[i] and surf[i] < lo:
            i_lo = i
            break
    if i_lo is None:
        return dict(a_lo=np.nan, a_hi=a_hi_x, width_arcmin=np.nan)
    a_lo_x = _interp_cross(i_lo, i_lo + 1, lo)
    return dict(a_lo=a_lo_x, a_hi=a_hi_x, width_arcmin=a_hi_x - a_lo_x)


# ─────────────────────────────── 汇总工具 ───────────────────────────────

def center_of(profile_r, n_rings=3):
    return float(np.nanmean(profile_r[:n_rings]))   # 同 forward_trace: 最内 3 环平均


def mags(x):
    return -2.5 * np.log10(x) if x > 0 else float("inf")


def stops(x):
    return float(np.log2(x)) if x > 0 else float("-inf")


# ─────────────────────────────── 模型卡 ───────────────────────────────

def run_model_card(n_rays=4_000_000, n_sun=2000, n_h_nodes=400, tau_steps=2000,
                   grid_half_km=9000.0, n_pix=360, n_r_bins=160, seed=0,
                   albedo_ratio=1.35, out_md=None):
    """默认参数(气溶胶+LD 开)跑全套, 写 docs/MODEL_CARD.md 并返回 metrics dict。"""
    import raytrace_eclipse as rte
    t0 = time.time()
    kw_phys = dict(aod550_trop=0.07, aod550_strat=0.005)

    print(f"[1/4] precompute_nodes(n_h={n_h_nodes}, tau_steps={tau_steps}, "
          f"aod={kw_phys['aod550_trop']}/{kw_phys['aod550_strat']}) ...")
    nodes = precompute_nodes(n_h_nodes, 90.0, 0.25, tau_steps, **kw_phys)
    bw = band_weights(nodes)
    wY = photopic_weights(nodes)
    channels = [dict(name=k, w=v["w"], dsc=v["dsc"]) for k, v in bw.items()]
    channels.append(dict(name="phot", w=wY, dsc=1.0))
    channels.append(dict(name="B2c", **mono_weights(nodes, 491.0)))
    channels.append(dict(name="B4c", **mono_weights(nodes, 654.0)))

    print(f"[2/4] 默认撒线 {n_rays:,} × n_sun={n_sun} (气溶胶+LD 开, "
          f"{len(channels)} channel × ext/geo) ... [{time.time()-t0:.0f}s]")
    res = scatter(nodes, channels, n_rays_b=n_rays, n_sun=n_sun,
                  grid_half_km=grid_half_km, n_pix=n_pix, n_r_bins=n_r_bins,
                  limb_dark=True, seed=seed, center_stats_band="V")

    # 分子大气上限: 同节点把气溶胶两组分置 0(α/柱密度不变), LD 关(对齐 -13.5 基线口径)
    print(f"[3/4] 分子上限撒线(aod=0, LD 关, phot only) ... [{time.time()-t0:.0f}s]")
    nodes_mol = dict(nodes, tau_t550=np.zeros_like(nodes["tau_t550"]),
                     tau_s550=np.zeros_like(nodes["tau_s550"]),
                     aod550_trop=0.0, aod550_strat=0.0)
    wY_mol = photopic_weights(nodes_mol)
    res_mol = scatter(nodes_mol, [dict(name="phot", w=wY_mol, dsc=1.0)],
                      n_rays_b=n_rays, n_sun=n_sun, grid_half_km=grid_half_km,
                      n_pix=n_pix, n_r_bins=n_r_bins, limb_dark=False, seed=seed)

    print(f"[4/4] raytrace_eclipse.forward_trace(默认+分子对照, {n_rays:,}) "
          f"取 raw sRGB R/B + 悬崖 ... [{time.time()-t0:.0f}s]")
    res_rt = rte.forward_trace(n_rays_b=n_rays, n_sun=n_sun,
                               grid_half_km=grid_half_km, h_direct_max=5500.0,
                               verbose=False)
    res_rt_mol = rte.forward_trace(n_rays_b=n_rays, n_sun=n_sun,
                                   grid_half_km=grid_half_km, h_direct_max=5500.0,
                                   aod550_trop=0.0, aod550_strat=0.0,
                                   limb_dark=False, verbose=False)

    # ---- 暗端 ----
    c_phot = center_of(res["prof"]["phot"]["ext"])
    c_phot_mol = center_of(res_mol["prof"]["phot"]["ext"])
    band_rows = {}
    for b in ("B", "V", "R", "I"):
        ce = center_of(res["prof"][b]["ext"])
        cg = center_of(res["prof"][b]["geo"])
        band_rows[b] = dict(total_mag=mags(ce), geo_mag=mags(cg),
                            text_mag=mags(ce / cg) if cg > 0 else np.nan)

    # ---- 绿松石带: 三口径 R/B ----
    rbp = rb_shu_profile(albedo_ratio=albedo_ratio, nodes=nodes, scatter_res=res,
                         verbose=False)
    a = rbp["a_arcmin"]
    surf_phot = res["prof"]["phot"]["ext"]
    ribbons = {k: measure_rb_ribbon(a, rbp[f"rb_{k}"], surf=surf_phot)
               for k in ("raw", "sun_norm", "shu", "shu_mono")}

    # raw sRGB R/B(渲染口径, forward_trace), 默认 + 分子对照
    def _srgb_min(r):
        rb_rt = np.asarray(r["RB_r"], float)
        surf_rt = np.asarray(r["surf_r"], float)
        a_rt = np.degrees(np.arctan(np.asarray(r["r_cent"]) / D_MOON)) * 60.0
        v_rt = np.isfinite(rb_rt) & (surf_rt > 0)
        i_rt = np.where(v_rt)[0][np.argmin(rb_rt[v_rt])]
        return float(rb_rt[i_rt]), float(a_rt[i_rt]), a_rt, surf_rt

    srgb_rb_min, srgb_rb_pos, a_rt, surf_rt = _srgb_min(res_rt)
    srgb_rb_min_mol, srgb_rb_pos_mol, _, _ = _srgb_min(res_rt_mol)

    # 单层(单条切高光线)窄带透射比的物理下限 vs 卷积后剖面实际凹陷深度
    ok_n = ~nodes["blocked"]
    w_b2c = mono_weights(nodes, 491.0)["w"]
    w_b4c = mono_weights(nodes, 654.0)["w"]
    with np.errstate(invalid="ignore", divide="ignore"):
        tr_nodes = np.where(w_b2c > 0, w_b4c / w_b2c, np.nan)
    i_fl = np.nanargmin(np.where(ok_n, tr_nodes, np.nan))
    single_layer = dict(floor=float(tr_nodes[i_fl]),
                        z_tan=float(nodes["z_tan"][i_fl]))
    snm = rbp["rb_sun_norm_mono"]
    v_dip = np.isfinite(snm) & (surf_phot > 1e-7) & (a > 30) & (a < 47)
    i_dip = np.where(v_dip)[0][np.argmin(snm[v_dip])]
    band_dip = dict(min=float(snm[i_dip]), a=float(a[i_dip]))

    cliff = cliff_width(a_rt, surf_rt)
    # 本影边缘最陡坡度(档/arcmin): log2 剖面 3-bin 平滑后取梯度极值(35-50' 窗口)
    with np.errstate(divide="ignore"):
        st_rt = np.log2(np.clip(surf_rt, 1e-12, None))
    win = np.isfinite(st_rt) & (a_rt > 35) & (a_rt < 50) & (surf_rt > 1e-9)
    if win.sum() > 4:
        sm = np.convolve(st_rt[win], np.ones(3) / 3, mode="same")
        slope_max = float(np.max(np.gradient(sm, a_rt[win])))
    else:
        slope_max = np.nan

    metrics = dict(
        n_rays=n_rays, n_sun=n_sun, n_h_nodes=n_h_nodes, tau_steps=tau_steps,
        grid_half_km=grid_half_km, n_pix=n_pix, n_r_bins=n_r_bins, seed=seed,
        aod550_trop=kw_phys["aod550_trop"], aod550_strat=kw_phys["aod550_strat"],
        center_stops_default=stops(c_phot), center_stops_mol=stops(c_phot_mol),
        center_stops_rt=float(res_rt["center_stops"]),
        band_rows=band_rows, sun_ratio=rbp["sun_ratio"],
        sun_ratio_mono=rbp["sun_ratio_mono"], albedo_ratio=albedo_ratio,
        ribbons=ribbons, srgb_rb_min=srgb_rb_min, srgb_rb_pos=srgb_rb_pos,
        srgb_rb_min_mol=srgb_rb_min_mol, srgb_rb_pos_mol=srgb_rb_pos_mol,
        single_layer=single_layer, band_dip=band_dip,
        cliff=cliff, cliff_slope_max=slope_max, runtime_s=time.time() - t0,
    )

    os.makedirs(OUTPUTS, exist_ok=True)
    np.savez(os.path.join(OUTPUTS, "model_card_run.npz"),
             r_cent=res["r_cent"], a_arcmin=a,
             **{f"{n}_{k}": res["prof"][n][k] for n in res["prof"]
                for k in ("ext", "geo")},
             phot_mol_ext=res_mol["prof"]["phot"]["ext"],
             rb_raw=rbp["rb_raw"], rb_sun_norm=rbp["rb_sun_norm"],
             rb_shu=rbp["rb_shu"], rb_sun_norm_mono=rbp["rb_sun_norm_mono"],
             rb_shu_mono=rbp["rb_shu_mono"],
             rt_r_cent=res_rt["r_cent"], rt_surf_r=res_rt["surf_r"],
             rt_RB_r=res_rt["RB_r"])

    md = _format_model_card(metrics)
    if out_md is None:
        out_md = os.path.join(DOCS, "MODEL_CARD.md")
    with open(out_md, "w") as f:
        f.write(md)
    print(md)
    print(f"\n已写入 {out_md}（用时 {metrics['runtime_s']:.0f}s）")
    return metrics


def _format_model_card(m):
    import datetime
    today = datetime.date.today().isoformat()
    br = m["band_rows"]
    rib = m["ribbons"]
    S = m["sun_ratio"]
    Sm = m["sun_ratio_mono"]
    A = m["albedo_ratio"]

    # Shu 口径裁决段: 数据驱动。核心判据 = 卷积后窄带凹陷深度(band_dip) vs 单层物理
    # 下限(single_layer)。凹陷不存活时如实记录与文献的张力, 不调参凑。
    sl = m["single_layer"]
    bd = m["band_dip"]
    dip_survives = bd["min"] < 0.97
    if dip_survives:
        shu_analysis = (
            f"**Shu 口径裁决**：窄带凹陷在卷积后存活（sun-norm λ_eff 最低 "
            f"{bd['min']:.3f} @ {bd['a']:.1f}'，单层物理下限 {sl['floor']:.3f} @ "
            f"z_tan≈{sl['z_tan']:.0f}km），Shu 口径数字见上表，与文献 in-band 0.8–1.0 "
            f"的吻合情况以上表为准。")
    else:
        shu_analysis = (
            f"**Shu 口径裁决（当前模型复现不出文献 ribbon，如实记录）**：单层窄带凹陷"
            f"存在（654/491 透射比最低 {sl['floor']:.3f} @ z_tan≈{sl['z_tan']:.0f}km，"
            f"臭氧 Chappuis 所致），但经 32' 太阳盘卷积 + 本影内深红层光流入后，1D 径向"
            f"平均剖面上**不存活**——带区窄带凹陷只剩 {bd['min']:.3f}（@ {bd['a']:.1f}'，"
            f"统计噪声量级），R/B 偏离基线 <1%。因此无论 boxcar 还是 λ_eff 变体，"
            f"R/B<1 判据完全由基线常数与 1 的相对位置决定（boxcar {S*A:.3f} 全程 ≥1 "
            f"无 ribbon；λ_eff {Sm*A:.3f} 基线本身 <1 而无界），没有物理 ribbon——"
            f"文献的 in-band 0.8–1.0 复现不出。蓝化本身在宽带渲染口径是稳健的"
            f"（min sRGB R/B {m['srgb_rb_min']:.2f}，分子口径 {m['srgb_rb_min_mol']:.2f}）："
            f"sRGB R 通道的有效权重正压在 Chappuis 核（~580–640nm），而 GF-4 B4"
            f"（630–690nm）大半避开了核区，对 teal 天然不敏感。与文献张力的候选归因，"
            f"按证据强度排序：(1) **z_tan 8–19km 红肩消光缺失**（对流层顶薄卷云/"
            f"subvisible cirrus + UTLS 增强消光，本模型未建模）：该层占带区通量 "
            f"47–83%，机制探针显示对其额外压暗 2–3 mag（垂直 OD ~0.04–0.07，"
            f"slant ×40–50）即让 λ_eff ribbon 从无到有（宽 76–79 km，与文献 "
            f"120–190 km 同量级偏窄）且中心档数纹丝不动；转正需独立证据——"
            f"CALIPSO/SPARC 卷云气候学 + 2019-01-21 当晚 limb 一圈云况；"
            f"(2) GF-4 PMS 实际光谱响应非 boxcar，若 B4 响应含 600–630nm 边带则直接"
            f"采到 Chappuis 核；(3) 文献是 2D 边界像素测量 + 月面纹理（月海 Ti）耦合，"
            f"与 1D 轴对称径向平均不是同一把尺（续16 已知口径差；点太阳对照下 1D 径向"
            f"折叠本身就抹掉凹陷）。"
            f"\n\n**三层解耦结论（2026-06-10 敏感性矩阵裁决）**：z_tan<8km 深层光只控"
            f"暗端（带区通量占比仅 0.3%，人工压暗 3 mag 把中心档数推进 Mallama 区间而 "
            f"dip 恢复量精确为零）；8–19km 红肩控 ribbon（见候选 (1)）；~20km 平流层层"
            f"只控宽带 teal 对比（sAOD 0→0.005 吃掉 sRGB 对比 0.047，对窄带 dip 仅 "
            f"0.0012）——三层互不串扰，暗端、ribbon、宽带 teal 是三个独立的开放项。")
    shu_analysis += "（按验证纪律如实记录，未调参凑文献。）"
    public_statement = (
        f"> 在仅含 Rayleigh+O3+背景气溶胶（参数取 2019-01 气候学观测值，未调参）的"
        f"模型下，宽带渲染口径的蓝化稳健（min sRGB R/B≈{m['srgb_rb_min']:.2f}，"
        f"分子上限 {m['srgb_rb_min_mol']:.2f}），但 Shu 2024 卫星窄带口径的 R/B<1 "
        f"ribbon 在 1D 径向平均上复现不出（凹陷 <1%，噪声级）。敏感性分析排除了背景"
        f"气溶胶参数与本影深层红光作为根因：带区通量的 47–83% 来自切高 8–19km 的"
        f"“红肩”光，文献量级的 ribbon 需要该层额外 ~2–3 mag 的 slant 消光"
        f"（垂直 OD ~0.02–0.07，在对流层顶薄卷云气候学范围内，但本模型未建模、亦无"
        f"当晚 limb 云况证据，故如实留作开放项）。另有 GF-4 光谱响应、月面反照率 "
        f"±15%、2D 边界测量 vs 1D 径向平均三个口径差，量级均足以改写“R/B<1”"
        f"判据的边界。")
    table_note = ("" if dip_survives else
                  "\n注：sun-norm 与 λ_eff 行的最蓝点/「<1 区间」是 ±0.3% 统计噪声"
                  "穿越各自基线所致，**无物理 ribbon**（见下方裁决段）。\n")

    def _rib_row(name, base, r):
        if r["bounded"]:
            pos = f"{r['inner']:.1f}–{r['outer']:.1f}'"
            wid = f"{r['width_arcmin']:.1f}' ≈ {r['width_km']:.0f} km"
            inband = f"{r['rb_min']:.3f}–1.0"
        elif np.isfinite(r["inner"]):
            pos, wid, inband = f"{r['inner']:.1f}'–∞（无界）", "无界", "无 ribbon 语义"
        elif np.isfinite(r["rb_min"]) and r["rb_min"] >= 1.0:
            pos, wid, inband = "无（全程 ≥1）", "—", "无 ribbon"
        else:
            pos, wid, inband = "—", "—", "—"
        return (f"| {name} | {base} | {r['rb_min']:.3f} @ {r['a_min']:.1f}' "
                f"| {inband} | {pos} | {wid} |")

    return f"""# MODEL_CARD — 当前模型关键数字（口径全标注）

> 由 `python src/band_profile.py --model-card` 生成，{today}。
> **物理改动后必须重跑本卡**（任何 src/ 物理模块变更都会使下面的数字过期）。

## 跑参

- 撒线 n_rays={m['n_rays']:,} × 太阳子点 n_sun={m['n_sun']}，seed={m['seed']}
- h 网格 n_h_nodes={m['n_h_nodes']}（RK4 ds=0.25km，τ 积分 tau_steps={m['tau_steps']}，
  消光用真实切点高度 z_tan 口径）
- 落点网格 ±{m['grid_half_km']:.0f} km / {m['n_pix']}px，径向 {m['n_r_bins']} bins
- 物理默认全开：背景气溶胶 AOD550 对流层 {m['aod550_trop']}（指数廓线 H=1.5km，
  α={atm.ALPHA_TROP}）+ 平流层 {m['aod550_strat']}（对流层顶锚定指数尾 z0=12km/H=6km，
  α={atm.ALPHA_STRAT}；k550(20km)=2.2e-4、k550(25km)=0.96e-4、sAOD550=0.005，
  三硬约束全过，Wrana/Thomason/Kloss 换算），太阳 limb darkening（Allen V 带二次律，均值归一）
- 消光成分：Rayleigh + O3 + 背景气溶胶。**未建模**：H2O/O2 吸收带（影响 I 带）、云、多次散射

## 暗端（本影中心，反日轴最内 3 环平均，0 档 = 未食满月）

| 量 | 数值 | 口径 |
|---|---|---|
| 本影中心 photopic（分子大气上限） | **{m['center_stops_mol']:.2f} 档** | aod=0、LD 关；纯 Rayleigh+O3 的理论亮度上限 |
| 本影中心 photopic（默认：气溶胶+LD） | **{m['center_stops_default']:.2f} 档** | 本卡默认物理 |
| 同口径 cross-check（forward_trace 全谱 Y） | {m['center_stops_rt']:.2f} 档 | 同参独立管线，应与上行一致到 ~0.2 档（中心 3 环 Poisson） |

带内 magnitudes-lost（中心，total = 几何稀释 + 消光；T_eff = ext/geo 分解出的纯消光项；
每带各自以该带满月 = 0 mag，即带内归一口径）：

| 带 | total (mag) | 几何稀释-only (mag) | 纯消光 T_eff (mag) |
|---|---|---|---|
| B (390–490nm) | {br['B']['total_mag']:.2f} | {br['B']['geo_mag']:.2f} | {br['B']['text_mag']:.2f} |
| V (500–600nm) | {br['V']['total_mag']:.2f} | {br['V']['geo_mag']:.2f} | {br['V']['text_mag']:.2f} |
| R (570–720nm) | {br['R']['total_mag']:.2f} | {br['R']['geo_mag']:.2f} | {br['R']['text_mag']:.2f} |

I 带（720–880nm）管线缺 H2O 720/820nm 与 O2 吸收带，系统性**偏亮**，绝对值不引用。
对照锚点：Mallama 2022（clear-sky 经验消光模型）本影中心 disk-resolved V −20.5 档、
disk-integrated ≈−18.7 档；差距归因见 working.md 2026-06-09 暗端对账（消光标定口径为主因）。

## 绿松石带

- **min raw sRGB R/B = {m['srgb_rb_min']:.3f} @ {m['srgb_rb_pos']:.1f}'**（默认：气溶胶+LD）
  （渲染口径：forward_trace 线性 sRGB 通道比 R/B，含真实太阳谱、无任何归一化；
  这是视频/图片管线实际呈现的颜色量）
- 分子大气对照（aod=0、LD 关）：min raw sRGB R/B = {m['srgb_rb_min_mol']:.3f} @ {m['srgb_rb_pos_mol']:.1f}'
  （与历史金标准/勘误1 的 ~0.70@41' 同口径；背景气溶胶的 Ångström 蓝坡 + LD
  把对比冲淡到 {m['srgb_rb_min']:.2f}）
- 亮度悬崖：photopic 面亮度从满月的 10% 升到 90% 宽
  **{m['cliff']['width_arcmin']:.1f}'**（{m['cliff']['a_lo']:.1f}' → {m['cliff']['a_hi']:.1f}'，
  即整个半影爬升段）；本影边缘最陡坡度 **{m['cliff_slope_max']:.1f} 档/'**（35–50' 窗，3-bin 平滑）

R/B 口径（Shu 2024 / GF-4 窄带 B2=450–520nm、B4=630–690nm。
S = 太阳谱带平均比 B4/B2 = {S:.3f}（boxcar 带积分口径；单色 654/491 = {Sm:.3f}），
A = 月面反照率比 654/491 = {A}，月海/高地随 Ti 含量 **±15% 系统不确定度**）：

| 口径 | 直射区基线 | 最蓝点 | in-band 范围 | R/B<1 区间 | 宽度 |
|---|---|---|---|---|---|
{_rib_row("raw（灰月面带平均辐亮度直除，含太阳斜率）", f"{S:.3f}", rib['raw'])}
{_rib_row("sun-norm（÷S，中性白=1）", "1.000", rib['sun_norm'])}
{_rib_row("Shu 口径（raw×A，卫星 radiance 直除，boxcar）", f"{S*A:.3f}", rib['shu'])}
{_rib_row(f"Shu 口径 λ_eff 单色变体（654/491，S={Sm:.3f}）", f"{Sm*A:.3f}", rib['shu_mono'])}
{table_note}
对照文献（Shu 2024，RemoteSens 16(22):4181，GF-4 实测）：in-band R/B 0.8–1.0，
边界 ribbon 径向全宽 120–190 km（其口径是 2D 月盘边界过渡带 + 亮度可测窗，
比我们 1D 轴对称"R/B<1 全区间"窄，见 working.md 续16 的口径差分析）。

{shu_analysis}

**公开表述（对外写作的口径，DECISION 2026-06-10 §4 措辞）**：

{public_statement}

## 口径警告：三种 R/B 不可混比

1. **raw sRGB R/B**（渲染口径）——人眼/照片呈现用。与任何窄带辐亮度比不可比；
   渲染管线**绝不做 sun-normalization**（那会篡改人眼物理）。
2. **raw B4/B2**（灰月面带平均辐亮度比）——模型原始输出，基线 {S:.2f}<1，
   "R/B<1"无界、无 ribbon 语义，单独引用会把绿松石带说宽到任意大。
3. **sun-norm B4/B2**（中性白=1）——量"大气透射的相对蓝化"，对齐卫星定标反射率类
   比较；续16 的 in-band 数字是这个口径。
4. **Shu 口径 = raw×A**（卫星 radiance 直除）——与 Shu 2024 的 R/B<1 ribbon、
   in-band 0.8–1.0 直接可比；含反照率系统项 ±15%（A 取 {A}）。band 建模有两个变体：
   boxcar 带积分（基线 S·A = {S*A:.2f}）与 λ_eff 单色 654/491（基线 {Sm*A:.2f}）。
   单色变体里太阳斜率与反照率红坡近似相消（{Sm:.3f}×{A}≈{Sm*A:.2f}），数值上
   Shu≈sun-norm——这就是勘误2 说的"续16 把它当定标反射率比是物理理由错、数值歪打
   正着"；boxcar 变体二者只相消到 8%。

生成耗时 {m['runtime_s']:.0f}s。复现：`source .venv/bin/activate && python src/band_profile.py --model-card --n-rays {m['n_rays']}`
"""


# ─────────────────────────────── CLI ───────────────────────────────

def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-card", action="store_true",
                    help="默认参数(气溶胶+LD 开)跑全套并写 docs/MODEL_CARD.md")
    ap.add_argument("--n-rays", type=int, default=4_000_000)
    ap.add_argument("--n-sun", type=int, default=2000)
    ap.add_argument("--n-h", type=int, default=400)
    ap.add_argument("--tau-steps", type=int, default=2000)
    ap.add_argument("--albedo-ratio", type=float, default=1.35)
    ap.add_argument("--no-aerosol", action="store_true", help="关背景气溶胶")
    ap.add_argument("--no-limb-dark", action="store_true", help="关太阳 limb darkening")
    args = ap.parse_args()

    if args.model_card:
        run_model_card(n_rays=args.n_rays, n_sun=args.n_sun, n_h_nodes=args.n_h,
                       tau_steps=args.tau_steps, albedo_ratio=args.albedo_ratio)
        return

    aod_t, aod_s = (0.0, 0.0) if args.no_aerosol else (0.07, 0.005)
    t0 = time.time()
    nodes = precompute_nodes(args.n_h, 90.0, 0.25, args.tau_steps,
                             aod550_trop=aod_t, aod550_strat=aod_s)
    bw = band_weights(nodes)
    wY = photopic_weights(nodes)
    channels = [dict(name=k, w=v["w"], dsc=v["dsc"]) for k, v in bw.items()]
    channels.append(dict(name="phot", w=wY, dsc=1.0))
    print(f"节点完成 {time.time()-t0:.0f}s; blocked={int(nodes['blocked'].sum())}"
          f"/{args.n_h}, h_graze={nodes['h_nodes'][~nodes['blocked']][0]:.2f}km")
    res = scatter(nodes, channels, n_rays_b=args.n_rays, n_sun=args.n_sun,
                  limb_dark=not args.no_limb_dark, center_stats_band="V")
    print(f"撒线完成 {time.time()-t0:.0f}s")

    rr = res["r_cent"] / R_UMBRA
    print("\nr/R_u   B_lost  V_lost  R_lost  I_lost  (mag, 带内归一)")
    for i in range(0, len(rr), 4):
        if rr[i] > 1.25:
            break
        row = [mags(res["prof"][b]["ext"][i]) for b in "BVRI"]
        print(f"{rr[i]:5.3f}  " + "  ".join(f"{x:6.2f}" for x in row))
    for b in ["B", "V", "R", "I", "phot"]:
        ce = center_of(res["prof"][b]["ext"])
        cg = center_of(res["prof"][b]["geo"])
        print(f"center {b}: total={mags(ce):.2f} mag ({stops(ce):.2f} 档) | "
              f"geo-only={mags(cg):.2f} mag | T_eff={mags(ce/cg) if cg>0 else float('nan'):.2f} mag")

    # B2/B4 已在 channels 里, 顺手给三口径 R/B 摘要(完整 ribbon 度量走 --model-card)
    S = sun_band_ratio("B4", "B2", per_nm=True)
    rb = rb_calibers(res["prof"]["B4"]["ext"], res["prof"]["B2"]["ext"], S,
                     args.albedo_ratio)
    a = np.degrees(np.arctan(res["r_cent"] / D_MOON)) * 60.0
    v = np.isfinite(rb["shu"]) & (res["prof"]["phot"]["ext"] > 1e-7)
    if v.any():
        i = np.where(v)[0][np.argmin(rb["shu"][v])]
        print(f"\nR/B 最蓝点 @ {a[i]:.1f}': raw={rb['raw'][i]:.3f} "
              f"sun-norm={rb['sun_norm'][i]:.3f} Shu口径={rb['shu'][i]:.3f} "
              f"(S={S:.3f}, A={args.albedo_ratio})")


if __name__ == "__main__":
    _main()
