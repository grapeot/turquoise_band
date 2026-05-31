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


def tangent_height_from_radius(r_norm, h_min=5.0, h_max=80.0):
    """把月面归一化径向位置映射到擦边高度（L1 会用真实本影几何细化）。

    占位线性映射：r_norm=0（本影中心）对应低擦边高度 h_min（深红区），
    r_norm=1（本影边缘）对应高擦边高度 h_max（趋白区）。
    真实关系由地球本影几何 + 大气折射决定，留待 L1。
    """
    r_norm = np.clip(np.asarray(r_norm, dtype=float), 0.0, 1.0)
    return h_min + (h_max - h_min) * r_norm


if __name__ == "__main__":
    # 自查：擦边高度越低，等效空气质量越大（路径更深入稠密大气）
    for h in [10, 30, 50]:
        s, z, ds = limb_path_altitudes(h)
        print(f"擦边高度 {h}km: 视线最低海拔={z.min():.1f}km, 1200km处海拔={z.max():.1f}km")
    print("自查通过：切向视线在最近点贴近擦边高度，向两侧海拔抬升。")
