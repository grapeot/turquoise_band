"""入射谱：大气外太阳辐照谱。默认用实测 SAO2010 (Chance & Kurucz 2010)，回退 5772K 黑体。"""
import os
import numpy as np

H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23
T_SUN = 5772.0  # 太阳有效温度 (K)

_RAW = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
SAO2010_PATH = os.path.join(_RAW, "sao2010_solref.dat")

_solar_interp = None


def planck(lam_nm, T=T_SUN):
    """普朗克谱辐射（任意一致单位，仅形状重要）。"""
    lam = np.asarray(lam_nm, dtype=float) * 1e-9
    return (2 * H * C**2 / lam**5) / (np.exp(H * C / (lam * KB * T)) - 1.0)


def load_solar_spectrum(path=SAO2010_PATH):
    """加载 SAO2010 实测太阳谱（AM0）。

    格式：4 行说明 + 数据，列1=真空波长nm，列3=辐照度 W/m²/nm（直接用）。
    SAO2010 是 0.01nm 高分辨率（含 Fraunhofer 线），加载后建插值器；用到的网格
    较粗时插值会自然平滑掉细线，对宽带颜色计算无碍。
    """
    global _solar_interp
    from scipy.interpolate import interp1d
    d = np.loadtxt(path, comments="C")          # 跳过 "Column ..." 文本头
    lam, irr = d[:, 0], d[:, 2]                  # 真空波长(nm), W/m²/nm
    vis = (lam >= 360) & (lam <= 820)            # 只留可见+边，加速
    _solar_interp = interp1d(lam[vis], irr[vis], bounds_error=False, fill_value=0.0)
    return _solar_interp


def solar_spectrum(lam_nm):
    return _solar_interp(np.asarray(lam_nm, float)) if _solar_interp is not None else planck(lam_nm)


# 模块导入时自动加载实测谱；失败则保留黑体
try:
    load_solar_spectrum()
except Exception as _e:
    print(f"[solar] SAO2010 加载失败，回退 5772K 黑体: {_e}")


if __name__ == "__main__":
    lam = np.linspace(380, 780, 401)
    s = solar_spectrum(lam)
    peak = lam[np.argmax(s)]
    print(f"太阳谱峰值波长 ≈ {peak:.0f} nm")
    # 对照黑体看蓝端差异（SOURCES 记 400nm 真实/黑体 ≈0.92）
    bb = planck(lam)
    j400 = np.argmin(np.abs(lam - 400)); j550 = np.argmin(np.abs(lam - 550))
    ratio = (s[j400] / s[j550]) / (bb[j400] / bb[j550])
    print(f"400nm/550nm 真实÷黑体比 = {ratio:.2f}（SOURCES 记 ~0.92，<1 即蓝端偏低）")
    assert _solar_interp is not None, "应加载到实测谱"
    print("自查通过：已用 SAO2010 实测太阳谱。")
