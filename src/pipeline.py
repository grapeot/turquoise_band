"""L0 闭环：扫描擦边高度，产出色相/亮度/sRGB 曲线，并自查红→青→白趋势。

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
    """扫一组擦边高度，返回每个高度的 (h, hue, Y, sRGB)。"""
    h = np.linspace(h_min, h_max, n_h)
    lam = np.linspace(lam_min, lam_max, n_lam)
    I = rt.emergent_spectrum(h, lam)  # (H, L)

    XYZ = np.array([col.spectrum_to_XYZ(lam, I[i]) for i in range(len(h))])  # (H,3)
    Y = XYZ[:, 1]
    hue = np.array([col.hue_angle(XYZ[i]) for i in range(len(h))])
    # sRGB：每个高度按自身亮度归一（看色相），另存一条保留相对亮度的
    rgb_hue = np.array([col.XYZ_to_sRGB(XYZ[i], normalize=max(Y[i], 1e-9)) for i in range(len(h))])
    Ymax = Y.max()
    rgb_lum = np.array([col.XYZ_to_sRGB(XYZ[i], normalize=Ymax) for i in range(len(h))])
    return h, hue, Y, rgb_hue, rgb_lum, lam, I


def plot(h, hue, Y, rgb_hue, rgb_lum):
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)

    # 1. 色相角 vs 擦边高度
    axes[0].plot(h, hue, color="k", lw=2)
    axes[0].set_ylabel("色相角 (°)\n0=红 90=黄 180=青绿 270=蓝")
    axes[0].set_title("月食绿松石带：颜色随擦边高度的变化（L0 v0，含臭氧 Chappuis 近似）")
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


def self_check(h, hue, Y):
    """断言定性趋势：低高度偏红、中段转青绿、亮度随高度单调升。"""
    print("\n=== 自查趋势 ===")
    valid = ~np.isnan(hue)
    # 找绿松石带：色相角进入青绿区(150-220°)的高度窗口
    teal_mask = valid & (hue > 130) & (hue < 230)
    red_mask = valid & ((hue < 50) | (hue > 330))

    if red_mask.any():
        print(f"红区擦边高度范围: {h[red_mask].min():.0f}–{h[red_mask].max():.0f} km")
    if teal_mask.any():
        print(f"青绿(绿松石)带擦边高度范围: {h[teal_mask].min():.0f}–{h[teal_mask].max():.0f} km")
    else:
        print("⚠ 未检测到青绿色相区间")

    # 亮度应随高度大体单调升
    mono = np.mean(np.diff(Y) > 0)
    print(f"亮度随高度上升的比例: {mono*100:.0f}%")

    checks = {
        "存在红区(低高度)": red_mask.any() and h[red_mask].mean() < h.mean(),
        "存在青绿带": teal_mask.any(),
        "青绿带高度高于红区": (teal_mask.any() and red_mask.any() and h[teal_mask].mean() > h[red_mask].mean()),
        "亮度大体随高度升": mono > 0.8,
    }
    for k, v in checks.items():
        print(f"  [{'✓' if v else '✗'}] {k}")
    return all(checks.values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args()

    h, hue, Y, rgb_hue, rgb_lum, lam, I = scan()
    plot(h, hue, Y, rgb_hue, rgb_lum)
    if args.self_check:
        ok = self_check(h, hue, Y)
        print(f"\n{'✓ L0 闭环自查通过' if ok else '✗ 趋势未完全满足，需检查物理参数'}")
