"""入射谱：大气外太阳辐照谱。真实谱（ASTM/Kurucz）待接入，先用 5772K 黑体。"""
import numpy as np

H = 6.62607015e-34
C = 2.99792458e8
KB = 1.380649e-23
T_SUN = 5772.0  # 太阳有效温度 (K)

_solar_interp = None


def planck(lam_nm, T=T_SUN):
    """普朗克谱辐射（任意一致单位，仅形状重要）。"""
    lam = np.asarray(lam_nm, dtype=float) * 1e-9
    return (2 * H * C**2 / lam**5) / (np.exp(H * C / (lam * KB * T)) - 1.0)


def load_solar_spectrum(path):
    """加载真实太阳谱。文件两列：波长(nm), 辐照度。"""
    global _solar_interp
    from scipy.interpolate import interp1d
    d = np.loadtxt(path)
    _solar_interp = interp1d(d[:, 0], d[:, 1], bounds_error=False, fill_value=0.0)


def solar_spectrum(lam_nm):
    return _solar_interp(np.asarray(lam_nm, float)) if _solar_interp is not None else planck(lam_nm)


if __name__ == "__main__":
    lam = np.linspace(380, 780, 401)
    s = solar_spectrum(lam)
    peak = lam[np.argmax(s)]
    print(f"太阳谱峰值波长 ≈ {peak:.0f} nm（5772K 黑体应在 ~500nm）")
    assert 450 < peak < 550, "黑体峰值应在可见光蓝绿段"
    print("自查通过。")
