"""大气模型：海拔 → 空气数密度、臭氧数密度。

v0 先用解析近似让管线跑通（标度高度指数空气廓线 + 高斯臭氧层），
等 workflow 下到 US Std Atm 1976 真实空气廓线和标准臭氧廓线后，
从 data/raw 加载替换，对比曲线确认近似没带偏物理。
"""
import numpy as np

# 海平面空气数密度 (cm^-3)
N0_AIR = 2.546899e19
# 空气标度高度 (km)，对流层近似
H_SCALE_AIR = 8.0


def n_air_exponential(z_km):
    """指数廓线空气数密度 (cm^-3)。z<0 截断为海平面值。"""
    z = np.maximum(np.asarray(z_km, dtype=float), 0.0)
    return N0_AIR * np.exp(-z / H_SCALE_AIR)


def n_o3_gaussian(z_km, peak_km=22.0, width_km=5.0, column_DU=300.0):
    """高斯近似的臭氧数密度廓线 (cm^-3)。

    臭氧层峰值约在 20-25km，宽度若干 km。归一到给定柱总量（Dobson Unit）。
    1 DU = 2.687e16 分子/cm^2（垂直柱）。
    """
    z = np.asarray(z_km, dtype=float)
    shape = np.exp(-0.5 * ((z - peak_km) / width_km) ** 2)
    # 垂直积分归一到 column_DU
    zz = np.linspace(0, 100, 2000)
    shape_zz = np.exp(-0.5 * ((zz - peak_km) / width_km) ** 2)
    col_shape = np.trapezoid(shape_zz, zz * 1e5)  # cm^-2 (每单位峰值密度)
    target_col = column_DU * 2.687e16          # cm^-2
    n_peak = target_col / col_shape
    return n_peak * shape


# ---- 真实廓线：AFGL US Standard，默认自动加载 ----
_air_interp = None
_o3_interp = None


def use_real_profiles():
    """加载 AFGL US Standard 真实廓线（空气 + 臭氧数密度）。"""
    global _air_interp, _o3_interp
    import data_loaders
    _air_interp, _o3_interp = data_loaders.load_atmosphere()


# 模块导入时自动尝试加载真实数据；失败则保留解析近似
try:
    use_real_profiles()
except Exception as _e:
    print(f"[atmosphere] 真实廓线加载失败，回退解析近似: {_e}")


def n_air(z_km):
    return _air_interp(np.asarray(z_km, float)) if _air_interp is not None else n_air_exponential(z_km)


def n_o3(z_km):
    return _o3_interp(np.asarray(z_km, float)) if _o3_interp is not None else n_o3_gaussian(z_km)


if __name__ == "__main__":
    z = np.array([0, 10, 22, 40, 60])
    print("海拔(km)  空气密度(cm^-3)  臭氧密度(cm^-3)")
    for zi in z:
        print(f"  {zi:>3}    {n_air(zi):.3e}     {n_o3(zi):.3e}")
    # 自查：臭氧在 ~22km 峰值，空气随高度单调下降
    assert n_o3(22) > n_o3(0) and n_o3(22) > n_o3(60), "臭氧应在平流层峰值"
    assert n_air(0) > n_air(60), "空气密度应随高度下降"
    print("自查通过：臭氧层峰值在平流层，空气密度随高度衰减。")
