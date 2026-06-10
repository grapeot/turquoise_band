"""band_profile 的口径与物理回归测试。

测的是口径定义必须成立的不变量, 不是实现细节:
- 太阳谱带比值: 单色 654/491 ≈0.737(catch 太阳谱加载/波长轴回归), boxcar 带平均 ≈0.80
- 气溶胶提取: tau_aer550 与 tau_curved 的 on−off 差严格一致, aod=0 时恒为 0
- 集成(slow): 直射区 sun-norm R/B→1(±2%, catch 归一化/通量守恒回归),
  绿松石带凹陷存在且位置合理

跑: source .venv/bin/activate && python -m pytest src/tests/test_band_profile.py -v
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import band_profile as bp
import curved_path as cp


# ── 太阳谱带比值(口径常数, catch 太阳谱加载回归) ─────────────────────────────
def test_sun_band_ratio():
    """B4/B2 太阳谱比值: 单色中心 654/491 应 ≈0.73-0.75(勘误2 引的 0.737);
    boxcar 带平均(卫星辐亮度口径)应 ≈0.78-0.82; 带积分(不除带宽)≈0.67-0.70。
    三个口径常数都钉住, 任何一个漂了都说明太阳谱加载/波长轴出了回归。"""
    mono = float(bp.SUN(654.0) / bp.SUN(491.0))
    assert 0.72 < mono < 0.76, f"单色 654/491={mono:.4f} 应≈0.737"
    per_nm = bp.sun_band_ratio("B4", "B2", per_nm=True)
    assert 0.78 < per_nm < 0.82, f"带平均 B4/B2={per_nm:.4f} 应≈0.80"
    integ = bp.sun_band_ratio("B4", "B2", per_nm=False)
    assert 0.67 < integ < 0.70, f"带积分 B4/B2={integ:.4f} 应≈0.685"


# ── 气溶胶路径光学厚度提取 ──────────────────────────────────────────────────
def test_aerosol_tau550_extraction():
    """precompute_nodes 的 tau_aer550 应与 tau_curved 的 (aod on)−(aod off)
    在 550nm 严格一致(同 z_tan 同步数), 且 aod=0 时恒为 0、随切点高度趋零。"""
    nodes = bp.precompute_nodes(n_h_nodes=10, h_max=30.0, trace_ds_km=0.5,
                                tau_steps=600)
    lam = np.array([550.0])
    for i in range(len(nodes["h_nodes"])):
        if nodes["blocked"][i]:
            continue
        z = float(nodes["z_tan"][i])
        t_on = cp.tau_curved(z, lam, z_top_km=90.0, n_steps=600,
                             aod550_trop=0.07, aod550_strat=0.005)[0][0]
        t_off = cp.tau_curved(z, lam, z_top_km=90.0, n_steps=600)[0][0]
        assert np.isclose(nodes["tau_aer550"][i], t_on - t_off, rtol=1e-10, atol=1e-12), \
            f"z_tan={z:.1f}km: 提取 {nodes['tau_aer550'][i]:.4f} != 直接差 {t_on-t_off:.4f}"
    # 深擦边是一阶项, 高空趋零(Junge 层以上)
    ok = ~nodes["blocked"]
    assert nodes["tau_aer550"][ok][0] > 0.5, "深擦边 tau_aer550 应 >0.5(一阶项)"
    nodes0 = bp.precompute_nodes(n_h_nodes=6, h_max=30.0, trace_ds_km=0.5,
                                 tau_steps=400, aod550_trop=0.0, aod550_strat=0.0)
    assert np.allclose(nodes0["tau_aer550"], 0.0, atol=1e-12), "aod=0 时应恒为 0"


# ── 集成: 撒线后的口径不变量(慢, 标记 slow) ──────────────────────────────────
@pytest.mark.slow
def test_rb_calibers_scatter():
    """小规模撒线的口径不变量:
    (1) 直射区(a=66-74') sun-norm R/B 应 =1±2%(归一化/通量守恒回归);
    (2) 带剖面大尺度单调: 本影内红(sun-norm≫1)→边缘外趋 1, 36-38' > 40-42' >
        44-46'≈1(注意: 单层窄带凹陷 ~0.87 经太阳盘卷积+红层流入后在径向平均上
        不存活, 剖面只从红侧单调逼近 1, 见 MODEL_CARD Shu 口径裁决);
    (3) 三口径只差常数因子(口径关系不许漂)。"""
    rbp = bp.rb_shu_profile(n_rays_b=1_000_000, n_sun=600, n_h_nodes=250,
                            tau_steps=1000, n_pix=240, n_r_bins=120,
                            verbose=False)
    a = rbp["a_arcmin"]
    sn = rbp["rb_sun_norm"]
    direct = (a > 66) & (a < 74) & np.isfinite(sn)
    assert direct.any()
    assert np.all(np.abs(sn[direct] - 1.0) < 0.02), \
        f"直射区 sun-norm 偏离 1 超 2%: {sn[direct]}"

    def seg_mean(lo, hi):
        m = (a > lo) & (a < hi) & np.isfinite(sn)
        assert m.any(), f"{lo}-{hi}' 无有效 bin"
        return float(np.mean(sn[m]))

    s_in, s_mid, s_out = seg_mean(36, 38), seg_mean(40, 42), seg_mean(44, 46)
    assert s_in > s_mid > s_out, \
        f"带剖面应从红侧单调逼近 1: {s_in:.2f} > {s_mid:.2f} > {s_out:.2f} 不成立"
    assert s_in > 1.5, f"本影内侧 36-38' 应明显偏红(>1.5), 实测 {s_in:.2f}"
    assert abs(s_out - 1.0) < 0.05, f"44-46' 应已接近 1(±5%), 实测 {s_out:.3f}"

    # 三口径只差常数因子(口径关系本身不许漂)
    w = np.isfinite(sn)
    assert np.allclose(rbp["rb_raw"][w], sn[w] * rbp["sun_ratio"], rtol=1e-12)
    assert np.allclose(rbp["rb_shu"][w],
                       sn[w] * rbp["sun_ratio"] * rbp["albedo_ratio"], rtol=1e-12)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
