"""视线几何：切向（limb）穿透路径，以及擦边高度 ↔ 月面径向位置的映射。

折射进地球本影的阳光是掠过大气边缘的。给定一条擦边高度为 h_t 的切向视线，
它在大气里走的路径很长（air mass factor 可达数十）。这里用球对称几何
解析地把"沿视线的弧长"映射到"该点的海拔高度"，用于后续沿视线积分柱密度。

v1 采用直线视线近似（不显式追踪折射弯曲）来计算消光谱；折射主要影响
h_t ↔ 月面径向位置的映射和总光程，留待 L1 校核时细化。
"""
import numpy as np

R_EARTH = 6371.0  # 地球平均半径 (km)


def limb_path_altitudes(h_tangent_km, s_max_km=1200.0, n_samples=2000):
    """给定擦边高度，返回沿切向视线的采样点 (弧长 s, 海拔 z)。

    几何：视线是一条与地心距离最近点为 (R_EARTH + h_tangent) 的直线。
    取最近点为 s=0，沿视线走弧长 s 时，离地心距离为
        r(s) = sqrt((R_EARTH + h_tangent)^2 + s^2)
    海拔 z(s) = r(s) - R_EARTH。
    视线对称，单边积分到 s_max，总光程 ×2。

    返回
    ----
    s : 弧长采样 (km)，从 0 到 s_max
    z : 对应海拔 (km)
    ds : 采样步长 (km)，标量
    """
    s = np.linspace(0.0, s_max_km, n_samples)
    r_min = R_EARTH + h_tangent_km
    r = np.sqrt(r_min**2 + s**2)
    z = r - R_EARTH
    ds = s[1] - s[0]
    return s, z, ds


def column_density(h_tangent_km, density_func, s_max_km=1200.0, n_samples=2000):
    """沿擦边高度为 h_tangent 的切向视线，积分某成分的柱密度。

    参数
    ----
    h_tangent_km : 擦边高度 (km)
    density_func : 可调用，z(km) -> 数密度 (cm^-3)
    返回柱密度 (cm^-2)。注意视线对称，单边积分 ×2。
    """
    s, z, ds = limb_path_altitudes(h_tangent_km, s_max_km, n_samples)
    n = density_func(z)              # cm^-3
    ds_cm = ds * 1e5                 # km -> cm
    col_half = np.sum(n) * ds_cm     # 单边
    return 2.0 * col_half


# ============ L1：折射几何 — 擦边高度 → 本影内角度/径向位置 ============
# 物理处方见 docs/L1_geometry.md（基于 Robinson 2022, arXiv:2112.08966）。
# 太阳光掠过地球大气 limb 被折射 α(h) 弯入本影；α(h) ∝ 局地密度 ∝ exp(-h/H)。
# 关键：α(0)=70 arcmin 已含切点两侧对称双段，不再 ×2。

ALPHA0_RAD = 0.0204          # 地表掠射折射偏转角 (rad) ≈ 70 arcmin
H_REFRAC_KM = 8.0            # 折射标度高度 (km)
D_MOON_KM = 3.84e5           # 地月距离 (km)
R_SUN_KM = 6.96e5            # 太阳半径 (km)
D_SUN_KM = 1.496e8          # 日地距离 (km)


def refraction_angle(h_km):
    """擦边高度 h → 折射偏转角 α(h) (rad)。α(h)=α0·exp(-h/H)，消色差。"""
    return ALPHA0_RAD * np.exp(-np.asarray(h_km, dtype=float) / H_REFRAC_KM)


def umbra_radius_km():
    """月球处地球本影半径 (km)。会聚影锥：R_u = R⊕ − (R_sun−R⊕)·d_moon/d_sun。"""
    return R_EARTH - (R_SUN_KM - R_EARTH) * D_MOON_KM / D_SUN_KM


def umbra_radius_arcmin():
    """本影半径的角半径 (arcmin)，从月球看。"""
    return np.degrees(np.arctan(umbra_radius_km() / D_MOON_KM)) * 60.0


def shadow_radius_signed_km(h_km):
    """[贴轴侧 limb] 擦边高度 h 的光线落在本影内距影锥轴的带符号位置 (km)。

    r(h) = (R⊕+h) − α(h)·d_moon
    这是太阳**贴影锥轴那一侧 limb** 发出、向轴弯的光线落点，描述**红核**的色温分拣
    与会聚（低 h 强折射 → 落本影深处 → 聚焦增亮）。复现 Robinson 2022 Table 3.1 的 d 列。
    注意：不要用它定位绿松石带——那条带在月盘外缘，由太阳**对侧 limb** 照亮，
    用 shadow_radius_opposite_limb()。详见 docs/L1_geometry.md 的"矛盾解开"。
    """
    h = np.asarray(h_km, dtype=float)
    return (R_EARTH + h) - refraction_angle(h) * D_MOON_KM


def shadow_radius_opposite_limb(h_km):
    """[对侧 limb] 绿松石带/月盘外缘的擦边高度 h → 距影锥轴位置 (km)。

    r(h) = R_umbra − (α(h)·d_moon − h)
    照亮月盘外缘绿松石带的光来自太阳**对侧 limb**，穿地球另一侧平流层臭氧层。
    高 h（弱折射）→ 贴本影几何边界 R_umbra；归一化基准用 R_umbra，不是 R⊕。
    h=25-40km → 38-41 arcmin，紧贴边界 41.2'，与 Mallama/Robinson 自洽。
    """
    h = np.asarray(h_km, dtype=float)
    return umbra_radius_km() - (refraction_angle(h) * D_MOON_KM - h)


def shadow_radius_norm(h_km):
    """归一化本影径向位置 r_norm = |r_opp(h)| / R_umbra，截断到 [0,1]。

    用对侧-limb 映射（月盘外缘视角）。0=本影中心，1=本影几何边界 R_umbra。
    高 h（绿松石带）→ r_norm→1（贴边界），与文献"绿松石带紧贴本影边界"自洽。
    """
    r = np.abs(shadow_radius_opposite_limb(h_km))
    return np.clip(r / umbra_radius_km(), 0.0, 1.0)


def shadow_radius_arcmin(h_km):
    """擦边高度 h 落点距本影中心的角距离 (arcmin)，从月球看。

    用对侧-limb 映射（月盘外缘视角），使绿松石带(h~25-40km)落在 38-41'，贴边界。
    """
    r = np.abs(shadow_radius_opposite_limb(h_km))
    return np.degrees(np.arctan(r / D_MOON_KM)) * 60.0


def axis_arcmin(h_km):
    """贴轴-limb 落点距本影中心的角距 (arcmin)。红核源（与 shadow_radius_arcmin 对称）。"""
    r = np.abs(shadow_radius_signed_km(h_km))
    return np.degrees(np.arctan(r / D_MOON_KM)) * 60.0


def focusing_factor(h_km, r_floor_km=150.0):
    """几何聚焦因子（相对亮度增益），∝ b·|dh/dr| / r。

    环带能量守恒：limb 环 [h,h+dh] 供光 ∝ 2π·b·dh，映射到本影环 [r,r+dr] 接收 ∝ 2π·r·dr。
    故亮度 ∝ b·|dh/dr|/r。低 h 光线被会聚到近中心(小 r)→ 中心更亮（红核虽消光重却最亮）。
    r_floor：太阳非点源对中心 1/r 发散的正则化软下限。
    """
    h = np.asarray(h_km, dtype=float)
    b = R_EARTH + h
    dr_dh = 1.0 + refraction_angle(h) * D_MOON_KM / H_REFRAC_KM   # dr/dh
    dh_dr = 1.0 / dr_dh
    r = np.maximum(np.abs(shadow_radius_signed_km(h)), r_floor_km)
    return b * dh_dr / r


# 向后兼容别名
def focusing_jacobian(h_km):
    return focusing_factor(h_km)


def tangent_height_from_radius(r_norm, h_min=5.0, h_max=80.0):
    """[已弃用] L0 占位线性映射，保留向后兼容。L1 改以 h 为自变量，见上方折射几何。"""
    r_norm = np.clip(np.asarray(r_norm, dtype=float), 0.0, 1.0)
    return h_min + (h_max - h_min) * r_norm


if __name__ == "__main__":
    # 自查：擦边高度越低，等效空气质量越大（路径更深入稠密大气）
    for h in [10, 30, 50]:
        s, z, ds = limb_path_altitudes(h)
        print(f"擦边高度 {h}km: 视线最低海拔={z.min():.1f}km, 1200km处海拔={z.max():.1f}km")
    print("自查通过：切向视线在最近点贴近擦边高度，向两侧海拔抬升。\n")

    # ---- L1 折射几何自查：对照 Robinson 2022 Table 3.1 的 d 列 ----
    print(f"本影半径: {umbra_radius_km():.0f} km = {umbra_radius_arcmin():.1f} arcmin")
    print("\n擦边高度  α(arcmin)  落点r(带符号,km)  Robinson_d  r_norm  角距(arcmin)")
    robinson_d = {0: -1620, 8: 3724, 18: 5604, 25: 6142, 32: 6338, 50: 6420}
    for h in [0, 8, 18, 25, 32, 50]:
        a_arcmin = np.degrees(refraction_angle(h)) * 60
        r_signed = shadow_radius_signed_km(h)
        rn = shadow_radius_norm(h)
        arcmin = shadow_radius_arcmin(h)
        print(f"  {h:>4}    {a_arcmin:>7.2f}    {r_signed:>10.0f}      {robinson_d[h]:>6}    {rn:.3f}   {arcmin:.1f}")

    # 验证高 h 端（绿松石带相关）与 Robinson 吻合
    err_50 = abs(shadow_radius_signed_km(50) - 6420) / 6420
    err_25 = abs(shadow_radius_signed_km(25) - 6142) / 6142
    print(f"\nh=50km 相对误差 {err_50*100:.1f}%, h=25km 相对误差 {err_25*100:.1f}%")
    assert err_50 < 0.05 and err_25 < 0.05, "高h端应与 Robinson Table 3.1 吻合(<5%)"
    # 方向：低 h → 小 r_norm（本影深处），高 h → r_norm→1（边缘）
    assert shadow_radius_norm(5) < shadow_radius_norm(40), "低h应落本影深处，高h落边缘"
    print("L1 折射几何自查通过：映射方向对，高h端与文献吻合。")
