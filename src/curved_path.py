"""真·弯曲光路上的消光积分 —— 替换 geometry.column_density 的直线近似。

物理图景
========
擦边高度 h 的太阳光擦过地球大气 limb, 被折射弯向地心。它**不是直线**: 折射
让它在最近点(切点)附近向下弯, 多停留在稠密低层大气里, 真实空气质量(air mass)
比同切高的直线弦更大 → 消光更强。直线近似系统性偏亮。

如何得到真实弯曲轨迹(无解析 α 处方, 第一性原理)
------------------------------------------------
球对称大气里折射率只依赖地心距 r: n(r)=1+(n_sea-1)·ρ(z)/ρ(0), z=r-R⊕,
ρ 用真实 AFGL 廓线(atmosphere.n_air)。光线满足 Bouguer 不变量(球对称 Snell):
    n(r)·r·sin(ψ) = const = L
ψ 是光线与局地径向的夹角。在切点 r_t(ψ=90°) 处 L=n(r_t)·r_t。
沿光路 r 从 r_t 单调增, 由不变量得
    sin(ψ) = L/(n(r)·r),  cos(ψ)=sqrt(1-sin^2 ψ)
弧长元 ds 与径向元 dr 的关系: dr/ds=cos(ψ) → ds=dr/cos(ψ)。
于是真实路径长度元 ds=dr/sqrt(1-[L/(n r)]^2)。这就是弯曲带来的额外 air mass:
当 ψ→90°(贴近切点)分母→0, 路径在切点附近被拉长(几何上正确, 光线在那里近水平
长距离穿行稠密层)。

光学厚度沿真实路径积分(切点两侧对称, ×2):
    τ(λ)=2 ∫_{r_t}^{r_top} [n_air(z)σ_ray(λ)+n_o3(z)σ_o3(λ)] · ds/dr · dr
与直线版本的唯一区别就是 ds/dr 用真实 1/cos(ψ) 而非直线的 r/sqrt(r^2-r_t^2)。
(直线 = 把 n 设为 1 的退化情形, 可由同一套代码 with_refraction=False 复现, 自洽对照。)

注意: 这里只算"给定切高 h 的一条光线的消光谱", 不碰落点几何。落点(focusing/
本影中心亮度)由 brute_ray_trace 撒线涌现, 与本模块解耦。本模块把它的透射谱喂给那条管线。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import atmosphere
import cross_sections

R_EARTH = 6371.0  # km

# 海平面空气折射率 (n-1)。可见光 ~600nm 干空气标准值 ≈ 2.78e-4。
# 与 cross_sections 里 Edlén 色散一致量级; 折射的色散(蓝光弯更多)由落点管线另算,
# 这里消光积分用消色差的几何路径(路径形状对 λ 的依赖 ~1e-4 量级, 对 τ 可忽略)。
N_SEA_REFRACTIVITY = 2.78e-4


def refractive_index(z_km):
    """折射率 n(z)=1+(n_sea-1)·ρ(z)/ρ(0)。ρ 用真实 AFGL 空气数密度廓线。"""
    rho_ratio = atmosphere.n_air(z_km) / atmosphere.n_air(0.0)
    return 1.0 + N_SEA_REFRACTIVITY * rho_ratio


def _path_factor(r_km, L, with_refraction=True):
    """ds/dr = 1/cos(ψ), 其中 sin(ψ)=L/(n(r)·r)。返回沿路径的几何拉伸因子。

    L 是 Bouguer 不变量 = n(r_t)·r_t。直线退化(with_refraction=False)时 n≡1,
    L=r_t, 还原 ds/dr=r/sqrt(r^2-r_t^2)(球面直线弦)。
    """
    z = r_km - R_EARTH
    n = refractive_index(z) if with_refraction else np.ones_like(np.asarray(r_km, float))
    sin_psi = L / (n * r_km)
    sin_psi = np.clip(sin_psi, -1.0, 1.0)
    cos_psi = np.sqrt(np.clip(1.0 - sin_psi**2, 1e-30, None))
    return 1.0 / cos_psi


def tau_curved(h_tangent_km, lam_nm, z_top_km=90.0, n_steps=20000,
               with_refraction=True):
    """沿真实弯曲光路积分光学厚度 τ(λ)。

    参数
    ----
    h_tangent_km : 切点高度(擦边高度) km
    lam_nm       : 波长数组 nm
    z_top_km     : 大气上界(此上消光可忽略) km
    with_refraction : True=真实弯曲路径; False=直线对照(n≡1)
    返回 τ(λ), 形状同 lam_nm。已含切点两侧对称(×2)。

    数值: 径向网格在切点附近用 sqrt 间距加密(那里 1/cos ψ 近奇异, 可积但需细)。
    """
    lam = np.asarray(lam_nm, dtype=float)
    r_t = R_EARTH + h_tangent_km
    r_top = R_EARTH + z_top_km

    # Bouguer 不变量: 切点 ψ=90° → L=n(r_t)·r_t
    n_t = refractive_index(h_tangent_km) if with_refraction else 1.0
    L = n_t * r_t

    # 切点附近 1/cos ψ ~ 1/sqrt(r-r_t) 型可积奇异。用变量替换 r=r_t+u^2 把奇点抹平,
    # 在 u 上均匀采样 → r 在切点附近自动加密, 积分收敛快且稳。
    u_max = np.sqrt(r_top - r_t)
    u = np.linspace(0.0, u_max, n_steps)
    r = r_t + u**2
    dr_du = 2.0 * u                      # dr/du

    z = r - R_EARTH
    pf = _path_factor(r, L, with_refraction=with_refraction)   # ds/dr

    # 各成分沿路径数密度
    n_air_path = atmosphere.n_air(z)     # cm^-3
    n_o3_path = atmosphere.n_o3(z)       # cm^-3

    # 被积函数(对 r): [n_air σ_ray + n_o3 σ_o3] · (ds/dr)。再乘 dr/du, 在 u 上积。
    # σ 依赖 λ → 把 air/o3 的"路径柱积分(对 u)"先算成标量, 再外积 σ。
    # ∫ n · (ds/dr) dr = ∫ n · pf · (dr/du) du
    integrand_air = n_air_path * pf * dr_du   # cm^-3 · km (×1e5→cm 后是柱密度被积)
    integrand_o3 = n_o3_path * pf * dr_du

    du = u[1] - u[0]
    km_to_cm = 1e5
    col_air_half = np.trapezoid(integrand_air, dx=du) * km_to_cm   # cm^-2 单边
    col_o3_half = np.trapezoid(integrand_o3, dx=du) * km_to_cm

    N_air = 2.0 * col_air_half           # 切点两侧对称
    N_o3 = 2.0 * col_o3_half

    sig_ray = cross_sections.sigma_rayleigh(lam)
    sig_o3 = cross_sections.sigma_o3(lam)
    tau = N_air * sig_ray + N_o3 * sig_o3
    return tau, N_air, N_o3


def transmission_curved(h_tangent_km, lam_nm, **kw):
    """透射率 exp(-τ) 沿真实弯曲路径。"""
    tau, _, _ = tau_curved(h_tangent_km, lam_nm, **kw)
    return np.exp(-tau)


def column_density_curved(h_tangent_km, species="air", z_top_km=90.0,
                          n_steps=20000, with_refraction=True):
    """沿真实弯曲路径的柱密度(cm^-2)。species ∈ {air,o3}。drop-in 替换 geometry.column_density。"""
    _, N_air, N_o3 = tau_curved(h_tangent_km, np.array([600.0]),
                                z_top_km=z_top_km, n_steps=n_steps,
                                with_refraction=with_refraction)
    return N_air if species == "air" else N_o3


if __name__ == "__main__":
    lam = np.array([450.0, 550.0, 650.0])
    j_red = 2
    print("=== 弯曲路径 vs 直线 光学厚度对照 (650nm 红光) ===")
    print("h(km)  τ_curved  τ_straight  比值   air质量比   弯曲衰减(mag)  直线衰减(mag)")
    for h in [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]:
        tc, Nair_c, _ = tau_curved(h, lam, with_refraction=True)
        ts, Nair_s, _ = tau_curved(h, lam, with_refraction=False)
        mag_c = 2.5 * np.log10(np.exp(tc[j_red]))
        mag_s = 2.5 * np.log10(np.exp(ts[j_red]))
        ratio = tc[j_red] / ts[j_red]
        am_ratio = Nair_c / Nair_s
        print(f" {h:>4.0f}   {tc[j_red]:7.3f}   {ts[j_red]:7.3f}  {ratio:5.2f}   "
              f"{am_ratio:5.2f}      {mag_c:6.2f}        {mag_s:6.2f}")

    # h=0 完整谱
    print("\n=== h=0 擦地光线弯曲路径透射谱 ===")
    lam_full = np.array([450.0, 500.0, 550.0, 600.0, 650.0, 700.0])
    tc, Nair, No3 = tau_curved(0.0, lam_full, with_refraction=True)
    print(f"h=0 弯曲: N_air={Nair:.3e} cm^-2, N_o3={No3:.3e} cm^-2")
    print("λ(nm)  τ_curved  T=exp(-τ)  衰减(mag)")
    for i, l in enumerate(lam_full):
        print(f" {l:.0f}   {tc[i]:7.3f}   {np.exp(-tc[i]):.3e}   {2.5*np.log10(np.exp(tc[i])):6.2f}")
