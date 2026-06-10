"""光度曲线: 本影中心→满月的 photopic 径向剖面 (真·正向 ray tracing)。

旧版 outputs/photometric_profile.png 的生成脚本散佚(2026-06-09 审计发现), 本脚本收编重建。
物理走权威管线 raytrace_eclipse.forward_trace(默认参数: 4M 光线 × 2000 太阳子点,
气溶胶两组分 + 太阳 limb darkening, 含半影直射光), 零解析处方。

输出 outputs/photometric_profile.png:
  上图: 相对光度(满月=1, log 纵轴)
  下图: 曝光档数(log2, 相对满月) + 关键区注释
        本影内血月缓升(擦边高度 0→30km 消光指数降) → 41.2' 绿松石带亮度悬崖(进半影,
        太阳探头) → 半影直射光平滑爬升 → 73' 满月 0 档。

用法: ./.venv/bin/python scripts/render_photometric_profile.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import geometry as g
import raytrace_eclipse as rte
import render as R

OUT_PATH = os.path.join(R.OUT, "photometric_profile.png")


def main():
    t0 = time.time()
    # 默认物理参数(4M 光线)。grid_half_km=9000 让径向剖面覆盖到半影外缘 73'+
    # (默认 7000 只到 ~63'), 与 build_lut_from_raytrace 的覆盖口径一致。
    res = rte.forward_trace(grid_half_km=9000.0, verbose=True)
    print(f"forward_trace 用时 {time.time() - t0:.0f}s")

    d_moon = g.D_MOON_KM
    a = np.degrees(np.arctan(np.asarray(res["r_cent"]) / d_moon)) * 60.0
    surf = np.asarray(res["surf_r"])
    ok = np.isfinite(surf) & (surf > 0)
    a, surf = a[ok], surf[ok]
    stops = np.log2(surf)

    a_umb = np.degrees(np.arctan(g.umbra_radius_km() / d_moon)) * 60.0   # ≈41.2'
    a_full = a_umb + 32.0                                                # 本影边界+太阳全径=满月

    # 关键数字(stdout 留档)
    center_stops = res["center_stops"]
    s73 = float(np.interp(min(a_full, a[-1]), a, surf))
    print(f"本影中心 {center_stops:.2f} 档; 73' 处相对光度 {s73:.3f} "
          f"({np.log2(s73):.2f} 档, 应≈0)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    def _zones(ax):
        ax.axvline(a_umb, color="gray", ls="--", lw=1.2, label=f"本影边界{a_umb:.0f}'")
        ax.axvline(a_full, color="green", ls=":", lw=1.2, label=f"半影外缘{a_full:.0f}'(满月)")
        ax.axvspan(38, 44, color="cyan", alpha=0.15, label="绿松石带")
        ax.axvspan(44, a_full, color="orange", alpha=0.08, label="半影区(直射光渐入)")

    # 上图: 相对光度(log 纵轴)
    _zones(ax1)
    ax1.semilogy(a, surf, color="crimson", lw=2)
    ax1.set_xlabel("距本影中心角距(arcmin)")
    ax1.set_ylabel("相对光度(满月=1)")
    ax1.set_title("月食光度剖面: 本影中心→满月 (真·光线追踪, 气溶胶两组分+太阳limb darkening)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3, which="both")

    # 下图: 曝光档数 + 关键区注释
    _zones(ax2)
    ax2.plot(a, stops, color="tab:blue", lw=2.5)
    ax2.set_xlabel("距本影中心角距(arcmin)")
    ax2.set_ylabel("曝光档数(相对满月)")
    ax2.set_title(f"曝光档数剖面 (本影中心{center_stops:.1f}档 → 满月0档, 平滑无跃变)")
    ax2.grid(alpha=0.3)
    ann = dict(color="#555", fontsize=9,
               arrowprops=dict(arrowstyle="->", color="#888"))
    ax2.annotate("本影中心\n古铜血月(缓升:擦边高度0→30km)", xy=(3, float(np.interp(3, a, stops))),
                 xytext=(5, center_stops + 2.5), **ann)
    ax2.annotate("绿松石带\n亮度悬崖", xy=(41, float(np.interp(41, a, stops))),
                 xytext=(33, -7.5), **ann)
    ax2.annotate("半影:直射光渐入(爬升)", xy=(57, float(np.interp(57, a, stops))),
                 xytext=(52, -3.8), **ann)
    ax2.annotate("满月", xy=(a_full, float(np.interp(a_full, a, stops))),
                 xytext=(a_full - 4, 1.0), **ann)
    ax2.set_ylim(center_stops - 2.5, 2.0)

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=120)
    plt.close(fig)
    print(f"已存: {OUT_PATH}")


if __name__ == "__main__":
    main()
