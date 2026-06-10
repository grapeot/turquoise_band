"""【解析对照/教学链 L0】闭环：扫描擦边高度，产出色相/亮度/sRGB 曲线，并自查红→青→白趋势。

用法：
    python src/pipeline.py            # 跑并存图到 outputs/
    python src/pipeline.py --self-check   # 加跑趋势断言
"""
import os
import sys
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import radiative_transfer as rt
import color as col
import geometry

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")


def scan(h_min=8.0, h_max=70.0, n_h=120, lam_min=380, lam_max=780, n_lam=401):
    """扫一组擦边高度，返回每个高度的 (h, hue, Y, sRGB)。

    关键（见 science review）：颜色归一化常数从未衰减入射日光算一次，全程复用——
    白点 white_XYZ = 入射日光的 XYZ。这样 Y(h) 自动携带真实消光导致的亮度暴跌，
    色相也都相对同一个白点参考，不做 per-height 归一。
    """
    h = np.linspace(h_min, h_max, n_h)
    lam = np.linspace(lam_min, lam_max, n_lam)

    # 未衰减入射日光的 XYZ —— 固定参考白点
    import solar
    I_sun = solar.solar_spectrum(lam)
    white_XYZ = col.spectrum_to_XYZ(lam, I_sun)
    k = 1.0 / white_XYZ[1]  # 让白点 Y=1

    I = rt.emergent_spectrum(h, lam)  # (H, L)
    XYZ = np.array([col.spectrum_to_XYZ(lam, I[i]) for i in range(len(h))]) * k  # (H,3)，同一 k
    white_XYZ_norm = white_XYZ * k  # Y=1

    # L1：聚焦因子乘到亮度（XYZ 整体缩放，不碰色相）
    foc = geometry.focusing_jacobian(h)            # (H,)
    foc = foc / foc.max()                          # 归一，仅看相对增亮
    XYZ_lum = XYZ * foc[:, None]

    Y = XYZ_lum[:, 1]
    hue = np.array([col.hue_angle(XYZ[i], white_XYZ=white_XYZ_norm) for i in range(len(h))])

    # sRGB：全程用同一白点归一，保留相对亮度（含聚焦，暗处自然变暗）
    rgb_lum = np.clip(np.array([col.XYZ_to_sRGB(XYZ_lum[i]) for i in range(len(h))]), 0, 1)
    # 另存一条：每高度按自身亮度提亮（只为看色相，不代表真实亮度）
    rgb_hue = np.array([col.XYZ_to_sRGB(XYZ[i], normalize=max(XYZ[i, 1], 1e-12)) for i in range(len(h))])

    # L1：每个擦边高度对应的本影内角度/径向位置
    arcmin = geometry.shadow_radius_arcmin(h)      # 距本影中心角距
    r_norm = geometry.shadow_radius_norm(h)
    Y_trans = XYZ[:, 1]                            # 仅透射(不含聚焦)的亮度，用于物理趋势自查
    rgb_hue = np.clip(rgb_hue, 0, 1)
    return h, hue, Y, rgb_hue, rgb_lum, lam, I, arcmin, r_norm, Y_trans


def plot(h, hue, Y, rgb_hue, rgb_lum):
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    # 1. 色相角 vs 擦边高度
    axes[0].plot(h, hue, color="k", lw=2)
    axes[0].set_ylabel("色相角 (°)\n0=红 90=黄 180=青绿 270=蓝")
    axes[0].set_title("月食绿松石带：颜色随擦边高度的变化（L0 v1，真实臭氧截面+AFGL大气）")
    axes[0].grid(alpha=0.3)
    axes[0].axhspan(0, 40, alpha=0.08, color="red")
    axes[0].axhspan(150, 200, alpha=0.08, color="teal")

    # 2. 相对亮度 vs 擦边高度（对数）
    axes[1].semilogy(h, np.maximum(Y, 1e-9), color="darkorange", lw=2)
    axes[1].set_ylabel("相对亮度 Y (对数)")
    axes[1].grid(alpha=0.3, which="both")

    # 3. 颜色条带：保留相对亮度的真实观感
    axes[2].imshow(rgb_lum[None, :, :], aspect="auto",
                   extent=[h[0], h[-1], 0, 1])
    axes[2].set_yticks([])
    axes[2].set_xlabel("擦边高度 (km)  ——  低=本影深处，高=本影边缘")
    axes[2].set_title("合成色带（保留相对亮度）：应呈 暗红 → 绿松石 → 白", fontsize=10)

    # 中文字体兜底
    for ax in axes:
        ax.title.set_fontsize(11)
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti"]
    plt.rcParams["axes.unicode_minus"] = False

    fig.tight_layout()
    p = os.path.join(OUT, "L0_hue_curve.png")
    fig.savefig(p, dpi=130)
    print(f"已存图：{p}")

    # 单独存一张纯色带
    fig2, ax2 = plt.subplots(figsize=(10, 1.5))
    ax2.imshow(rgb_lum[None, :, :], aspect="auto", extent=[h[0], h[-1], 0, 1])
    ax2.set_yticks([])
    ax2.set_xlabel("擦边高度 (km)")
    fig2.tight_layout()
    p2 = os.path.join(OUT, "L0_colorband.png")
    fig2.savefig(p2, dpi=130)
    print(f"已存色带：{p2}")
    return p


def plot_angular(h, hue, Y, rgb_lum, arcmin, r_norm):
    """L1 主图：横轴 = 投影到月面的角度（距本影中心 arcmin），双轴标 r_norm。

    以 h 为内部采样、但画在角度坐标上，正确反映绿松石带挤在本影边缘的"窄"。
    """
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti"]
    plt.rcParams["axes.unicode_minus"] = False

    # 按角距排序（h→arcmin 单调，但保险起见排一下）
    order = np.argsort(arcmin)
    x = arcmin[order]
    hue_s, Y_s, rgb_s, rn_s, h_s = hue[order], Y[order], rgb_lum[order], r_norm[order], h[order]
    R_u = geometry.umbra_radius_arcmin()

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    # 1. 色相角
    axes[0].plot(x, hue_s, color="k", lw=2)
    axes[0].set_ylabel("色相角 (°)\n0=红 90=黄 180=青绿 270=蓝")
    axes[0].set_title("月食绿松石带：颜色 vs 月面角位置（L1，含折射几何映射）")
    axes[0].grid(alpha=0.3)
    axes[0].axhspan(0, 50, alpha=0.08, color="red")
    axes[0].axhspan(150, 240, alpha=0.08, color="teal")
    axes[0].axvline(R_u, color="gray", ls="--", lw=1)
    axes[0].text(R_u, axes[0].get_ylim()[1]*0.9, f" 本影几何边界 {R_u:.0f}'",
                 fontsize=8, color="gray")

    # 2. 亮度（对数，含聚焦）
    axes[1].semilogy(x, np.maximum(Y_s, 1e-9), color="darkorange", lw=2)
    axes[1].set_ylabel("相对亮度 Y (对数, 含聚焦)")
    axes[1].grid(alpha=0.3, which="both")
    axes[1].axvline(R_u, color="gray", ls="--", lw=1)

    # 3. 合成色带（角度坐标）
    axes[2].imshow(rgb_s[None, :, :], aspect="auto", extent=[x[0], x[-1], 0, 1])
    axes[2].set_yticks([])
    axes[2].set_xlabel("距本影中心角距 (arcmin)  ——  小=本影深处(红核)，大=边缘(绿松石/白)")
    axes[2].set_title("合成色带（角度坐标，保留相对亮度）", fontsize=10)
    axes[2].axvline(R_u, color="white", ls="--", lw=1, alpha=0.7)

    # 顶轴：r_norm 刻度
    def arcmin_to_rnorm(a):
        return np.interp(a, x, rn_s)
    secax = axes[0].secondary_xaxis("top")
    ticks_rn = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]
    tick_arcmin = [np.interp(t, rn_s, x) for t in ticks_rn]
    secax.set_xticks(tick_arcmin)
    secax.set_xticklabels([f"{t:.2f}" for t in ticks_rn])
    secax.set_xlabel("归一化本影半径 r/R_umbra (0=中心, 1=边缘)")

    for ax in axes:
        ax.title.set_fontsize(11)
    fig.tight_layout()
    p = os.path.join(OUT, "L1_angular_curve.png")
    fig.savefig(p, dpi=130)
    print(f"已存 L1 角度图：{p}")

    # 单独色带
    fig2, ax2 = plt.subplots(figsize=(10, 1.6))
    ax2.imshow(rgb_s[None, :, :], aspect="auto", extent=[x[0], x[-1], 0, 1])
    ax2.set_yticks([])
    ax2.axvline(R_u, color="white", ls="--", lw=1, alpha=0.7)
    ax2.set_xlabel("距本影中心角距 (arcmin)")
    fig2.tight_layout()
    p2 = os.path.join(OUT, "L1_colorband.png")
    fig2.savefig(p2, dpi=130)
    print(f"已存 L1 色带：{p2}")
    return p


def plot(h, hue, Y, rgb_hue, rgb_lum):
    """L0 诊断图：横轴擦边高度（保留作物理诊断）。"""
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    axes[0].plot(h, hue, color="k", lw=2)
    axes[0].set_ylabel("色相角 (°)")
    axes[0].set_title("诊断：颜色 vs 擦边高度（中间物理量）")
    axes[0].grid(alpha=0.3)
    axes[1].semilogy(h, np.maximum(Y, 1e-9), color="darkorange", lw=2)
    axes[1].set_ylabel("相对亮度 Y")
    axes[1].grid(alpha=0.3, which="both")
    axes[2].imshow(rgb_lum[None, :, :], aspect="auto", extent=[h[0], h[-1], 0, 1])
    axes[2].set_yticks([])
    axes[2].set_xlabel("擦边高度 (km)")
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti"]
    plt.rcParams["axes.unicode_minus"] = False
    fig.tight_layout()
    p = os.path.join(OUT, "diag_tangent_height.png")
    fig.savefig(p, dpi=130)
    print(f"已存诊断图：{p}")
    return p


def self_check(h, hue, Y, rgb_color, arcmin=None, Y_trans=None):
    """断言定性趋势。用 sRGB 实际通道关系判断红/青/白（比 Lab 色相角阈值更诚实）。

    - 红区：R 通道明显大于 B（暖色），出现在低擦边高度。
    - 青绿带：B 通道 >= R 通道（冷色），且非接近白（R<0.9 或 饱和度够）。
    - 白区：R,G,B 都接近 1。
    亮度趋势用透射亮度 Y_trans（不含聚焦），因为聚焦会让本影中心反而变亮。
    """
    print("\n=== 自查趋势 ===")
    if Y_trans is None:
        Y_trans = Y
    # 用 per-height 提亮的 rgb 判色相（rgb_lum 在本影深处压成纯黑，无法判色）
    R, G, B = rgb_color[:, 0], rgb_color[:, 1], rgb_color[:, 2]

    red_mask = (R > B + 0.08) & (R > G)                 # 暖：红/橙占优
    near_white = (R > 0.9) & (G > 0.9) & (B > 0.9)
    teal_mask = (B >= R - 0.02) & (~near_white) & (G > 0.2)  # 冷：青蓝，未到白

    if red_mask.any():
        print(f"暖色(红/橙)区擦边高度: {h[red_mask].min():.0f}–{h[red_mask].max():.0f} km, "
              f"亮度 {Y[red_mask].min():.1e}–{Y[red_mask].max():.1e}")
    if teal_mask.any():
        print(f"冷色(绿松石)带擦边高度: {h[teal_mask].min():.0f}–{h[teal_mask].max():.0f} km")
    if near_white.any():
        print(f"趋白区擦边高度: {h[near_white].min():.0f}–{h[near_white].max():.0f} km")

    # 角度坐标下报告绿松石带位置（这才是可观测坐标）
    if arcmin is not None and teal_mask.any():
        print(f"绿松石带角位置: {arcmin[teal_mask].min():.1f}–{arcmin[teal_mask].max():.1f} arcmin "
              f"(本影边界 {geometry.umbra_radius_arcmin():.1f}')")
        # 用 sRGB 红蓝比判偏蓝程度（比 CIELAB hue 角更诚实，见 LOG.md C 线诊断）
        tb = teal_mask
        rb_ratio = (R[tb].mean() + 1e-6) / (B[tb].mean() + 1e-6)
        print(f"绿松石带 sRGB 红蓝比 R/B = {rb_ratio:.2f} (文献微偏蓝 0.8-1.0；<0.8 偏蓝偏多)")

    mono = np.mean(np.diff(Y_trans) > 0)
    print(f"透射亮度随高度上升的比例: {mono*100:.0f}%")
    # 亮度跨度：本影深处应比边缘暗几个数量级
    dyn = Y_trans.max() / max(Y_trans.min(), 1e-12)
    print(f"透射亮度动态范围(边缘/深处): {dyn:.1e} 倍")

    checks = {
        "存在暖色红区(低高度)": red_mask.any() and h[red_mask].mean() < h.mean(),
        "存在绿松石冷色带": teal_mask.any(),
        "绿松石带高于红区": (teal_mask.any() and red_mask.any() and h[teal_mask].mean() > h[red_mask].mean()),
        "存在趋白区(高高度)": near_white.any() and h[near_white].mean() > h.mean(),
        "透射亮度大体随高度升": mono > 0.8,
        "透射亮度暴跌≥100倍": dyn > 1e2,
    }
    for k, v in checks.items():
        print(f"  [{'✓' if v else '✗'}] {k}")
    return all(checks.values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    # 以 h 为内部采样（绿松石带在角度上挤在边缘，按 h 密采才采得到）
    h, hue, Y, rgb_hue, rgb_lum, lam, I, arcmin, r_norm, Y_trans = scan(h_min=2.0, h_max=70.0, n_h=160)
    plot_angular(h, hue, Y, rgb_lum, arcmin, r_norm)   # L1 主图：角度横轴
    plot(h, hue, Y, rgb_hue, rgb_lum)                  # 诊断图：擦边高度横轴
    if args.self_check:
        ok = self_check(h, hue, Y, rgb_hue, arcmin=arcmin, Y_trans=Y_trans)
        print(f"\n{'✓ L1 闭环自查通过' if ok else '✗ 趋势未完全满足，需检查物理参数'}")
