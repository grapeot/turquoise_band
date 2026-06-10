"""【共享基础】真实数据加载：把 data/raw 的权威数据解析成插值函数，接入管线。

- 臭氧截面：Serdyuchenko 2014，取平流层温度列（默认 233K）
- 大气廓线：AFGL US Standard，空气数密度 + 臭氧数密度（VMR→数密度）
"""
import os
import numpy as np
from scipy.interpolate import interp1d

RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Serdyuchenko 列顺序（0-based，列0=波长）：293,283,273,263,253,243,233,223,213,203,193 K
_O3_TEMP_COLS = {293: 1, 283: 2, 273: 3, 263: 4, 253: 5, 243: 6,
                 233: 7, 223: 8, 213: 9, 203: 10, 193: 11}


def load_o3_cross_section(temp_K=233, path=None):
    """加载 Serdyuchenko 臭氧截面，返回 σ(λnm)→cm² 的插值器。

    平流层有效温度约 220-250K，默认取 233K 列。
    """
    path = path or os.path.join(RAW, "o3_serdyuchenko_2014.dat")
    col = _O3_TEMP_COLS[temp_K]
    # 跳过 45 行文本头；某些温度列在部分波长缺测会是负值/占位，过滤掉
    data = np.loadtxt(path, skiprows=45, usecols=(0, col))
    lam, sig = data[:, 0], data[:, 1]
    mask = np.isfinite(sig) & (sig >= 0)
    lam, sig = lam[mask], sig[mask]
    # 只保留可见+近紫外范围加速插值
    vis = (lam >= 350) & (lam <= 830)
    lam, sig = lam[vis], sig[vis]
    order = np.argsort(lam)
    return interp1d(lam[order], sig[order], bounds_error=False, fill_value=0.0)


def load_atmosphere(path=None):
    """加载 AFGL US Standard，返回 (n_air_interp, n_o3_interp)，单位 cm⁻³，输入海拔 km。"""
    path = path or os.path.join(RAW, "afgl_us_standard.csv")
    d = np.genfromtxt(path, delimiter=",", names=True)
    z = d["z"]                       # km
    n_air = d["n"]                   # cm^-3
    o3_vmr_ppmv = d["O3"]            # ppmv
    n_o3 = o3_vmr_ppmv * 1e-6 * n_air  # cm^-3
    air_i = interp1d(z, n_air, bounds_error=False, fill_value=(n_air[0], 0.0))
    o3_i = interp1d(z, n_o3, bounds_error=False, fill_value=(n_o3[0], 0.0))
    return air_i, o3_i


if __name__ == "__main__":
    sig = load_o3_cross_section(233)
    print("臭氧截面@603nm =", f"{sig(603):.3e}", "cm² (文献峰值 ~5.2e-21)")
    print("臭氧截面@450nm =", f"{sig(450):.3e}", "cm² (蓝端应小很多)")
    assert sig(603) > sig(450) * 3, "Chappuis 峰应远高于蓝端"

    air_i, o3_i = load_atmosphere()
    print(f"空气数密度@0km = {air_i(0):.3e} cm⁻³ (应 ~2.5e19)")
    z_grid = np.linspace(0, 60, 200)
    o3_peak_z = z_grid[np.argmax(o3_i(z_grid))]
    print(f"臭氧数密度峰值高度 ≈ {o3_peak_z:.0f} km (应在平流层 20-35km)")
    assert 18 < o3_peak_z < 40, "臭氧峰应在平流层"
    print("自查通过：真实数据加载正确。")
