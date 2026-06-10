"""渲染链快速冒烟测试：权威 LUT 构建 + render_textured 默认引擎。

不渲大图、不跑 4M 光线——小参数验证管线"接得通且基本性质成立"：
- build_lut_from_raytrace: a 轴单调、覆盖 0-73'、XYZ 无 NaN/负值、外缘回到满月量级
- render_textured: 默认引擎是 raytrace；注入小 LUT 渲 96px 小图出有效像素
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import raytrace_eclipse as rte
import render_textured as rtx


@pytest.fixture(scope="module")
def small_lut():
    """小参数权威 LUT(~1-2s)。物理同默认管线, 只缩采样量。"""
    return rte.build_lut_from_raytrace(n_rays_b=300_000, n_sun=64,
                                       n_h_nodes=120, n_pix=120, n_disp=4)


def test_lut_monotonic_coverage_no_nan(small_lut):
    """LUT a 轴严格单调、覆盖 0-73'(首 bin 中心<2'、末端>72')、XYZ 无 NaN/负值。"""
    a, XYZ = small_lut["a"], small_lut["XYZ"]
    assert np.all(np.diff(a) > 0), "LUT 角距轴必须严格单调(插值前提)"
    assert a[0] < 2.0, f"LUT 内缘 {a[0]:.2f}' 应接近本影中心(<2')"
    assert a[-1] > 72.0, f"LUT 外缘 {a[-1]:.2f}' 应覆盖到满月端(>72')"
    assert np.isfinite(XYZ).all(), "LUT XYZ 不应有 NaN/Inf"
    assert (XYZ >= 0).all(), "LUT XYZ 不应有负值"


def test_lut_dark_center_bright_edge(small_lut):
    """物理量级: 本影中心远暗于满月端, 外缘亮度回到 ~1(直射光覆盖, 无采样截断假衰减)。"""
    Y = small_lut["XYZ"][:, 1]
    assert Y[0] < 1e-3, f"本影中心 Y={Y[0]:.2e} 应暗 ~10 档以上"
    assert 0.9 < Y[-1] < 1.1, f"满月端 Y={Y[-1]:.3f} 应回到 ~1.0"


def test_render_textured_default_engine_is_raytrace():
    """import 级冒烟: 默认引擎/默认 d 已切到真 ray tracing 物理。"""
    import inspect
    sig = inspect.signature(rtx.render_realistic_disk)
    assert sig.parameters["engine"].default == "raytrace"
    assert sig.parameters["d_arcmin"].default == 40.0


def test_render_textured_smoke_small(small_lut):
    """注入小 LUT 渲 96px 小图: 出 uint8、月盘有非零像素、引擎记录正确。"""
    rgb8, info = rtx.render_realistic_disk(size=96, ssaa=1, lut=small_lut,
                                           add_starfield=False, add_grain=False)
    assert rgb8.shape == (96, 96, 3) and rgb8.dtype == np.uint8
    assert info["engine"] == "raytrace"
    inside = info["inside"]
    # SSAA=1 时 inside 与输出同分辨率, 月盘内应有有效(非全黑)像素
    assert rgb8[inside].max() > 30, "月盘内应有可见像素(着色/曝光链没断)"


def test_render_textured_pointsource_legacy_smoke():
    """legacy 点源引擎(已废物理, 仅历史对照)仍可构建着色器并出有限值。"""
    shade_fn, a_lo, tables = rtx.build_shader(engine="pointsource")
    XYZ = shade_fn(np.array([30.0, 45.0, 60.0]))
    assert np.isfinite(XYZ).all() and a_lo >= 0.0
