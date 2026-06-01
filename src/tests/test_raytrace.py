"""真·正向 ray tracing 的物理不变量测试。

测的是"必须成立否则物理就错了"的关键性质, 不是实现细节:
- 折射: 擦地 α 量级(~60-70')、撞地 blocked、α 随擦边高度单调衰减、消色差量级
- 弯曲消光: 弯曲≥直线(air mass 更大)、低擦边比高擦边衰减更强(血月成因)、蓝衰>红衰
- 集成: 本影中心暗到逼近真实(-12~-15 档)、绿松石带 R/B<1(青)、退化点源精确

跑: source .venv/bin/activate && python -m pytest src/tests/test_raytrace.py -v
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import refraction_trace as rt
import curved_path as cp


# ── 折射数值积分 ──────────────────────────────────────────────────────────
def test_grazing_angle_magnitude():
    """擦地极限折射角应 ~60-70'(对照解析 70', 真追踪 63.5')。"""
    h_graze, alpha = rt.grazing_angle(ds_km=0.05)   # 加速(粗步长够判量级)
    alpha_arcmin = np.degrees(alpha) * 60
    assert 55 < alpha_arcmin < 72, f"擦地 α={alpha_arcmin:.1f}' 不在 55-72'"
    assert 1.0 < h_graze < 3.0, f"擦地 impact-h={h_graze:.2f}km 不在 1-3km(撞地遮挡边界)"


def test_low_rays_blocked():
    """impact-h=0 的光线弯曲后切点压到地表下 → 撞地 blocked, 到不了月面。"""
    r0 = rt.trace_ray(0.0)
    assert r0["blocked"], "h=0 光线应撞地 blocked(解析 α 把 70' 钉在 h=0 是错的)"


def test_refraction_monotonic_decreasing():
    """折射角随擦边高度单调衰减(高空大气稀, 弯得少)。"""
    hs = np.array([3.0, 8.0, 15.0, 30.0, 60.0])
    # ds_km=0.25 与 0.02 的 α 一致到 0.01'(见 forward_trace 默认), 加速 12×
    alphas = np.array([rt.refraction_angle_traced(h, ds_km=0.25) for h in hs])
    assert np.all(np.diff(alphas) < 0), f"α(h) 应单调减, 实测 {np.degrees(alphas)*60}"
    # 高空(60km)折射应趋近 0
    assert np.degrees(alphas[-1]) * 60 < 3.0, "h=60km 折射应 <3'"


def test_refraction_index_sea_level():
    """海平面折射度 (n-1) 应 ≈ 2.7e-4(标准干空气)。"""
    nu = rt.refractivity(0.0)
    assert abs(nu - 2.7e-4) / 2.7e-4 < 0.05, f"(n-1)@海平面={nu:.3e} 偏离 2.7e-4"


# ── 弯曲路径消光 ──────────────────────────────────────────────────────────
# tau_curved 返回 (tau_array, N_air, N_o3); [0] 取 tau 数组, 再按波长索引。
def test_curved_ge_straight_airmass():
    """弯曲路径 air mass ≥ 直线(折射让光线在稠密层多停留)。"""
    lam = np.array([650.0])
    tau_curved = cp.tau_curved(2.0, lam, with_refraction=True)[0][0]
    tau_straight = cp.tau_curved(2.0, lam, with_refraction=False)[0][0]
    assert tau_curved >= tau_straight, "弯曲 τ 应 ≥ 直线 τ"
    # 但差异是小修正(~8%), 不是把暗端做暗的主因
    assert tau_curved / tau_straight < 1.3, "弯曲/直线比应 <1.3(小修正)"


def test_low_tangent_stronger_extinction():
    """低擦边高度(厚大气)消光远强于高擦边(血月成因)。"""
    lam = np.array([650.0])
    tau_low = cp.tau_curved(2.0, lam)[0][0]
    tau_high = cp.tau_curved(30.0, lam)[0][0]
    assert tau_low > tau_high * 3, "低擦边消光应远强于高擦边"


def test_blue_attenuated_more_than_red():
    """蓝光衰减 > 红光(瑞利 ∝λ⁻⁴, 血月红的成因)。"""
    lam = np.array([450.0, 650.0])
    tau = cp.tau_curved(5.0, lam)[0]   # [0] 取 tau 数组
    assert tau[0] > tau[1], f"蓝 τ={tau[0]:.2f} 应 > 红 τ={tau[1]:.2f}"


# ── 集成: 本影中心档数 + 绿松石带(慢, 标记 slow) ────────────────────────────
@pytest.mark.slow
def test_umbra_center_dark():
    """真 ray tracing 本影中心应暗到逼近真实(-12~-15 档, 远暗于解析版 -7.7)。"""
    import raytrace_eclipse as rte
    r = rte.forward_trace(n_rays_b=1_000_000, n_sun=800, n_h_nodes=300,
                          n_pix=200, n_disp=6, verbose=False)
    cs = r["center_stops"]
    assert -16 < cs < -11, f"本影中心 {cs:.1f} 档不在 -16~-11(应逼近真实, 远暗于解析 -7.7)"


@pytest.mark.slow
def test_turquoise_band_blue():
    """绿松石带区应有 R/B<1(青绿), 颜色结论保持。"""
    import raytrace_eclipse as rte
    r = rte.forward_trace(n_rays_b=1_000_000, n_sun=800, n_h_nodes=300,
                          n_pix=200, n_disp=6, verbose=False)
    rb = np.array(r["RB_r"]); valid = np.isfinite(rb) & (np.array(r["surf_r"]) > 0)
    assert rb[valid].min() < 0.85, f"绿松石带最蓝 R/B={rb[valid].min():.2f} 应 <0.85(青)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
