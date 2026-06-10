"""【共享基础】大气模型：海拔 → 空气数密度、臭氧数密度、气溶胶消光廓线。

默认在模块导入时自动加载 AFGL US Standard 真实廓线（data_loaders，空气 + 臭氧），
解析近似（指数空气廓线 + 高斯臭氧层）仅作加载失败时的回退。气溶胶为两组分
（对流层 α=0.7 + 平流层对流层顶锚定指数尾 α=2.0，见 2026-06-10 换代记录）。
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


# 气溶胶 Ångström 指数, 分层(2026-06-10 裁决换代, 替代旧单一 α=1.3):
# - 对流层 0.7±0.4: limb 环以海洋气溶胶为主, 远洋粗粒子 α≈0.3、大陆 1.0-1.5
#   (MAN 船测, Smirnov 2011)。
# - 平流层 2.0±0.4: 硫酸盐细粒子, SAGE III 525/1020 消光比 3.2-4.8 → α 1.75-2.36;
#   Kloss 2020 pristine 期 AE≈1.7。
ALPHA_TROP = 0.7
ALPHA_STRAT = 2.0


def beta_aerosol_550_components(z_km, aod550_trop=0.07, h_trop_km=1.5,
                                aod550_strat=0.005, z0_strat_km=12.0,
                                h_strat_km=6.0):
    """背景气溶胶 550nm 垂直消光系数两组分 (beta_t, beta_s) (km⁻¹)。

    两组分波长依赖不同(对流层 ALPHA_TROP=0.7 / 平流层 ALPHA_STRAT=2.0), 必须
    分开返回、由调用方各自乘 (λ/550)^(−α) 再相加——这是 2026-06-10 参数换代的
    核心改动(旧版单一 α=1.3 合并返回)。

    对流层: 指数廓线 (aod_t/H)·exp(−z/H), H≈1.5km——边界层+自由对流层的"最晴夜"
    海洋背景, τa(500)≈0.07(MAN 船测, Smirnov 2011; 区间 0.04-0.10)。对带区是二阶
    (z>10km 处 <1e-4 km⁻¹)。

    平流层: 对流层顶锚定指数尾(2026-06-10 裁决, 替代旧高斯 z=20km/σ=2.5km):
        β_s(z) = (aod_s/H)·exp(−(z−z0)/H), z≥z0=12km, H=6km
    旧高斯把全柱挤进 19-23km, k550(20km)=8e-4 超观测 3-8×, 恰压绿松石带区。
    指数尾三硬约束全过(火山静默期/Ambae 衰减中段背景, 2019-01 口径):
      - k550(20km) = 2.2e-4 km⁻¹ ∈ (1–2.5)e-4   (Wrana 2021 / Thomason 2021 换算)
      - k550(25km) = 0.96e-4 km⁻¹ ∈ (0.5–1.5)e-4 (同上)
      - sAOD550 = 0.005 ∈ 0.004–0.006            (Kloss 2020 换算; "静默期
        0.001-0.003"系 1999-2004 极小期/1020nm 口径, 不适用于 2019-01)
    同时贴合"层主体贴对流层顶(中高纬 10-16km)、20km 以上递减"的观测形态
    (Thomason / Malinina 2021 / Wrana 2021)。
    简化注记: z0 以下硬截断是中高纬对流层顶的简化, z0 处 β 不连续(跳变量级
    8e-4 km⁻¹ = aod_s/H), 对 slant 积分是可接受的量级近似。
    """
    z = np.asarray(z_km, dtype=float)
    beta_t = (aod550_trop / h_trop_km) * np.exp(-np.clip(z, 0.0, None) / h_trop_km)
    beta_s = np.where(z >= z0_strat_km,
                      (aod550_strat / h_strat_km)
                      * np.exp(-(z - z0_strat_km) / h_strat_km), 0.0)
    return beta_t, beta_s


def beta_aerosol_550(z_km, aod550_trop=0.07, h_trop_km=1.5,
                     aod550_strat=0.005, z0_strat_km=12.0, h_strat_km=6.0):
    """两组分之和(对流层指数 + 平流层指数尾), km⁻¹。垂直积分 = aod_t + aod_s。

    注意: 合并值只适用于不区分波长依赖的诊断(如垂直 AOD 核算); 消光计算应使用
    beta_aerosol_550_components 两组分各配 ALPHA_TROP/ALPHA_STRAT。
    参数依据见 beta_aerosol_550_components docstring。
    """
    beta_t, beta_s = beta_aerosol_550_components(
        z_km, aod550_trop=aod550_trop, h_trop_km=h_trop_km,
        aod550_strat=aod550_strat, z0_strat_km=z0_strat_km,
        h_strat_km=h_strat_km)
    return beta_t + beta_s
