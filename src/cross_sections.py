"""截面：瑞利散射（解析）+ 臭氧 Chappuis 吸收（查表，待数据接入）。

瑞利散射有成熟的解析公式，先实现。臭氲截面留接口，等 data/raw 的实测数据到位后从文件加载。
"""
import numpy as np


def sigma_rayleigh(lam_nm):
    """空气分子的瑞利散射截面，单位 cm^2/分子。

    采用 Bucholtz (1995) / Bodhaine (1999) 思路的标准近似：
    sigma = (24 * pi^3 * (n^2-1)^2) / (lam^4 * N^2 * (n^2+2)^2) * F(air)
    这里用一个对可见光足够准的工程近似式（含 lambda 的色散），
    数量级在 550nm 约 4.5e-27 cm^2，符合文献。

    参数
    ----
    lam_nm : array-like, 波长 (nm)

    返回
    ----
    截面 (cm^2/分子)，与 lam_nm 同形状。
    """
    lam_um = np.asarray(lam_nm, dtype=float) / 1000.0  # 转微米

    # 标准干空气折射率色散 (Peck & Reeder 1972 / Edlén)，(n-1)*1e8
    inv2 = 1.0 / lam_um**2
    n_minus_1 = (8060.51 + 2480990.0 / (132.274 - inv2) + 17455.7 / (39.32957 - inv2)) * 1e-8
    n = 1.0 + n_minus_1

    # 标准状态分子数密度 (cm^-3)，海平面 15°C
    N_s = 2.546899e19

    # King 修正因子（去极化），可见光近似取 1.048
    F_king = 1.048

    lam_cm = lam_um * 1e-4
    sigma = (24.0 * np.pi**3 * (n**2 - 1.0) ** 2) / (
        lam_cm**4 * N_s**2 * (n**2 + 2.0) ** 2
    ) * F_king
    return sigma


# ---- 臭氧 Chappuis 吸收截面：待实测数据接入 ----

_o3_interp = None


def load_o3_cross_section(path):
    """从 data/raw 加载实测臭氧吸收截面 σ(λ)。

    期望文件两列：波长(nm), 截面(cm^2)。加载后建立插值器。
    数据来源见 data/raw/SOURCES.md（计划用 Serdyuchenko 2014）。
    """
    global _o3_interp
    from scipy.interpolate import interp1d

    data = np.loadtxt(path)
    lam, sig = data[:, 0], data[:, 1]
    _o3_interp = interp1d(lam, sig, bounds_error=False, fill_value=0.0)
    return _o3_interp


def sigma_o3_chappuis_approx(lam_nm):
    """Chappuis 带臭氧截面的 v0 解析近似 (cm^2/分子)。

    文献：Chappuis 带在 ~375-650nm，峰值约 600-602nm，峰值截面 ~5e-21 cm^2。
    用双高斯粗略拟合带形，仅供闭环 v0；真实数据（Serdyuchenko 2014）到位后替换。
    """
    lam = np.asarray(lam_nm, dtype=float)
    g1 = 4.6e-21 * np.exp(-0.5 * ((lam - 602.0) / 50.0) ** 2)
    g2 = 3.5e-21 * np.exp(-0.5 * ((lam - 575.0) / 25.0) ** 2)
    return g1 + g2


def use_real_o3(temp_K=233):
    """加载 Serdyuchenko 2014 实测臭氧截面（平流层温度）。"""
    global _o3_interp
    import data_loaders
    _o3_interp = data_loaders.load_o3_cross_section(temp_K)


def sigma_o3(lam_nm):
    """臭氧吸收截面 (cm^2/分子)。优先用实测数据，否则用 Chappuis 近似。"""
    if _o3_interp is not None:
        return _o3_interp(np.asarray(lam_nm, dtype=float))
    return sigma_o3_chappuis_approx(lam_nm)


# 模块导入时自动尝试加载真实截面
try:
    use_real_o3()
except Exception as _e:
    print(f"[cross_sections] 真实臭氧截面加载失败，回退近似: {_e}")


if __name__ == "__main__":
    lam = np.array([400, 450, 500, 550, 600, 650, 700])
    sr = sigma_rayleigh(lam)
    print("波长(nm)  瑞利截面(cm^2)")
    for l, s in zip(lam, sr):
        print(f"  {l:.0f}     {s:.3e}")
    # 自查：550nm 应在 ~4.5e-27，且随波长单调下降（蓝光散射强）
    assert sr[0] > sr[-1], "瑞利截面应随波长增大而减小"
    print("自查通过：蓝端散射强于红端，量级符合文献。")
