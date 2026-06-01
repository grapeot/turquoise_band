"""核心辐射传输：组装光学厚度 τ(λ,h)，算透射率与出射谱。

沿擦边高度为 h 的切向视线：
  τ(λ,h) = N_air(h)·σ_rayleigh(λ) + N_O3(h)·σ_O3(λ)
其中 N_x(h) 是该成分沿视线的柱密度 (cm^-2)，σ 是截面 (cm^2)。
瑞利当作消光（散射出视线即损失），臭氧当作吸收。单次散射近似。
全程 numpy 矩阵化：一次算 (h × λ) 的透射率矩阵。
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import atmosphere
import cross_sections
import solar
import geometry


def transmission_matrix(h_tangent_km, lam_nm, rayleigh=True, ozone=True, gray=False):
    """返回透射率矩阵 T，形状 (len(h), len(lam))。T[i,j] = exp(-τ(lam_j, h_i))。

    ablation 开关(默认全开=现状物理):
      rayleigh: 瑞利散射消光 on/off
      ozone:    臭氧 Chappuis 吸收 on/off
      gray:     灰消光模式——把瑞利+臭氧换成不分波长的灰消光(同总光学厚度均摊到所有λ),
                用于"消光但无颜色选择"的教学态(本影内变灰而非变红/青)。
    """
    h_arr = np.atleast_1d(np.asarray(h_tangent_km, dtype=float))
    lam = np.asarray(lam_nm, dtype=float)

    sig_ray = cross_sections.sigma_rayleigh(lam) if rayleigh else np.zeros_like(lam)
    sig_o3 = cross_sections.sigma_o3(lam) if ozone else np.zeros_like(lam)

    # 每个擦边高度的柱密度（对视线对称积分）
    N_air = np.array([geometry.column_density(h, atmosphere.n_air) for h in h_arr])  # (H,)
    N_o3 = np.array([geometry.column_density(h, atmosphere.n_o3) for h in h_arr])    # (H,)

    tau = np.outer(N_air, sig_ray) + np.outer(N_o3, sig_o3)  # (H, L)
    if gray:
        # 灰消光: 用各h的波长平均光学厚度替换, 消去颜色选择(消光但无色相)
        tau = np.tile(tau.mean(axis=1, keepdims=True), (1, len(lam)))
    return np.exp(-tau)


def emergent_spectrum(h_tangent_km, lam_nm, rayleigh=True, ozone=True, gray=False,
                      solar_mode="real"):
    """出射谱 I(λ,h) = I_sun(λ) · T(λ,h)，形状 (H, L)。

    ablation 开关(默认=现状物理): rayleigh/ozone/gray 见 transmission_matrix;
    solar_mode: "real"(实测/默认) 或 "blackbody"(5772K 黑体, 去太阳谱细结构的对照)。
    """
    lam = np.asarray(lam_nm, dtype=float)
    T = transmission_matrix(h_tangent_km, lam, rayleigh=rayleigh, ozone=ozone, gray=gray)
    if solar_mode == "blackbody":
        I_sun = solar.blackbody_spectrum(lam) if hasattr(solar, "blackbody_spectrum") \
            else solar.solar_spectrum(lam)
    else:
        I_sun = solar.solar_spectrum(lam)
    return I_sun[None, :] * T


if __name__ == "__main__":
    lam = np.linspace(380, 780, 401)
    h = np.array([10.0, 25.0, 50.0])
    T = transmission_matrix(h, lam)
    print("擦边高度  T@450nm(蓝)  T@600nm(橙)  T@680nm(红)")
    for i, hi in enumerate(h):
        j450 = np.argmin(np.abs(lam - 450))
        j600 = np.argmin(np.abs(lam - 600))
        j680 = np.argmin(np.abs(lam - 680))
        print(f"  {hi:>4}km   {T[i,j450]:.3e}   {T[i,j600]:.3f}      {T[i,j680]:.3f}")
    # 自查：低擦边高度蓝光几乎全灭（瑞利），红光相对透得过 → 红
    assert T[0, np.argmin(np.abs(lam-450))] < T[0, np.argmin(np.abs(lam-680))], "低高度处蓝应比红衰减强"
    print("自查通过：低擦边高度蓝光衰减远强于红光（血月成因）。")
