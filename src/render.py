"""【解析对照/教学链 + 共享显示链】L2 月盘 2D 渲染：把 L1 的 1D 径向物理（角距 → 颜色/亮度）贴到一张月盘图上。

注意：本模块的 LUT 渲染是教学链（解析 L1 物理）；其显示链（_tone_map_on_Y /
_xyz_to_srgb_linear / _srgb_gamma / _box_downsample 等）与几何常数被全项目渲染应用复用。

设计文档要点（v1，两个先拍板的决策）：

1) 角距坐标口径 —— 采用「口径 B」。
   shadow_radius_arcmin(h) 在 h→大 时趋向 R⊕ 对应的 57.6'，远超本影几何角半径 41.2'。
   这 57.6' 不是「本影边界」，而是这套 transmission-only 模型里折射光能落到的最外缘。
   若照实把 57.6' 当真实角距贴月盘（口径 A），月盘（半径 15.5'）采不到红核、绿松石被推出本影。
   故把模型角度结构线性压缩进真实本影：a_render = arcmin_model / arcmin_max × 41.2'。
   红核中心仍在 0'，模型最外缘对齐本影边界 41.2'。这是渲染层的视觉校准，不改物理管线。

2) tone-mapping 曝光 —— 全盘统一曝光，禁止 per-pixel 自适应。
   本影内亮度跨约 1.5万倍。全局曝光标量 E + Reinhard 在亮度通道上压缩，
   保住「中心暗红、边缘亮」的相对亮度结构。per-pixel 自动曝光会把红核拉到中灰，破坏观感。

辐射传输复用 radiative_transfer.emergent_spectrum，颜色复用 color，几何复用 geometry，
本模块只做 LUT 组装 + 月盘几何 + 显示链，不重新实现物理。

用法：
    python src/render.py                 # 默认 d=22', 1024px，出 moon_disk.png + 对照图
    python src/render.py --d 18 --size 1536
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
import solar

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")

# ---- 几何常数（arcmin），实测值见设计文档附表 ----
R_MOON_ARCMIN = np.degrees(np.arctan(1737.4 / 384400.0)) * 60.0   # 月盘视半径 ≈ 15.5'
R_UMBRA_ARCMIN = geometry.umbra_radius_arcmin()                   # 本影角半径 ≈ 41.2'


# ============================================================
# 1. 角距 → 线性 XYZ / Y 的 LUT（口径 B，边缘密采）
# ============================================================
def build_lut(n_h=600, h_min=2.0, h_max=70.0, lam_min=380, lam_max=780, n_lam=401,
              dense_lo=15.0, dense_hi=30.0, dense_extra=400, extra_dense_segs=None):
    """构建 1D LUT：渲染角距 a_lut(arcmin) → 线性 XYZ_lum、线性 Y。

    复用 rt.emergent_spectrum（辐射传输）与 col.spectrum_to_XYZ（颜色），与 pipeline.scan 同口径：
      - 固定白点 = 未衰减入射日光 XYZ，令白点 Y=1（不做 per-height 归一）。
      - 聚焦因子乘到亮度（只缩放 XYZ，不碰色相）。
    口径 B：建表轴预先压缩 a_lut = arcmin_model / arcmin_model.max() × R_UMBRA_ARCMIN。

    绿松石带在 h≈18–25km 对应角度极窄，在 [dense_lo, dense_hi] km 额外加密采样，
    保证绿松石带在最终角距轴上有足够采样点（避免色阶断层）。

    返回
    ----
    a_lut    : (N,) 升序渲染角距 (arcmin)，范围约 [0, 41.2]
    XYZ_lum  : (N,3) 线性 XYZ（含聚焦因子，未做曝光/gamma）
    Y_lum    : (N,) 线性亮度标量（= XYZ_lum[:,1]，单独存便于 log 域插值/tone-map）
    meta     : dict，附带 h、hue、arcmin_model 等用于自查
    """
    # 以 h 为内部采样变量，绿松石带所在 h 段加密
    h_uniform = np.linspace(h_min, h_max, n_h)
    h_dense = np.linspace(dense_lo, dense_hi, dense_extra)
    segs = [h_uniform, h_dense]
    # 额外加密段（如红核所在的低擦边高度段），消除强曝光下小角距区的色阶台阶
    for lo, hi, n in (extra_dense_segs or []):
        segs.append(np.linspace(lo, hi, n))
    h = np.unique(np.concatenate(segs))

    lam = np.linspace(lam_min, lam_max, n_lam)

    # 固定参考白点：未衰减入射日光，令白点 Y=1
    I_sun = solar.solar_spectrum(lam)
    white_XYZ = col.spectrum_to_XYZ(lam, I_sun)
    k = 1.0 / white_XYZ[1]
    white_XYZ_norm = white_XYZ * k

    # 辐射传输 → 出射谱 → 线性 XYZ（同一 k）
    I = rt.emergent_spectrum(h, lam)                                   # (H, L)
    XYZ = np.array([col.spectrum_to_XYZ(lam, I[i]) for i in range(len(h))]) * k  # (H,3)

    # L1 聚焦因子（相对增亮），只缩放亮度不碰色相
    foc = geometry.focusing_jacobian(h)
    foc = foc / foc.max()
    XYZ_lum = XYZ * foc[:, None]

    # 角距：几何已修正（对侧-limb 公式），绿松石带自然落 34-41' 本影内，
    # 不再需要口径 B 压缩——直接用真实角距。
    arcmin_model = geometry.shadow_radius_arcmin(h)
    a_lut_raw = arcmin_model

    # 按角距升序排
    order = np.argsort(a_lut_raw)
    a_lut = a_lut_raw[order]
    XYZ_lum = XYZ_lum[order]
    XYZ_hue = XYZ[order]              # 不含聚焦，用于判色相
    h_s = h[order]
    arcmin_model_s = arcmin_model[order]
    Y_lum = XYZ_lum[:, 1]

    # 色相角（用不含聚焦的 XYZ，避免聚焦把红核压暗影响判读）
    hue = np.array([col.hue_angle(XYZ_hue[i], white_XYZ=white_XYZ_norm)
                    for i in range(len(h_s))])

    meta = dict(h=h_s, hue=hue, arcmin_model=arcmin_model_s, white_XYZ_norm=white_XYZ_norm,
                arcmin_max=float(arcmin_model.max()))
    return a_lut, XYZ_lum, Y_lum, meta


def lookup_xyz(a, a_lut, XYZ_lum, Y_lum):
    """对像素角距数组 a，查 LUT 得线性 XYZ。

    - XYZ 的 X、Z 通道线性插值；亮度 Y 跨 1.5万倍动态范围，在 log 域插值避免暗区台阶。
    - 端点钳到首尾（a< a_lut[0] 取首，a> a_lut[-1] 取尾）。
    返回 (..., 3) 线性 XYZ。
    """
    a = np.asarray(a, dtype=float)
    # 色度：用 XYZ 归一到 Y=1 的色度向量插值，再乘回插值后的 Y，保证色相平滑且亮度走 log 域
    Y_ref = np.maximum(Y_lum, 1e-30)
    chroma = XYZ_lum / Y_ref[:, None]                 # (N,3)，Y 通道≈1
    cX = np.interp(a, a_lut, chroma[:, 0])
    cZ = np.interp(a, a_lut, chroma[:, 2])
    logY = np.interp(a, a_lut, np.log(Y_ref))
    Y = np.exp(logY)
    X = cX * Y
    Z = cZ * Y
    return np.stack([X, Y, Z], axis=-1)


# ============================================================
# 2. 月盘几何 + 渲染
# ============================================================
def render_disk(d_arcmin=26.0, size=1024, margin_arcmin=4.0, ssaa=2,
                exposure=None, target_srgb=0.80, lut_kwargs=None):
    """渲染一张月盘 PNG（线性 XYZ → 全局曝光 → Reinhard → sRGB → 8bit）。

    坐标系：以本影中心 O_umbra 为原点，月盘中心 O_moon 在 +x 方向距离 d 处。
    画幅覆盖月盘 [d-R_moon-margin, d+R_moon+margin]，正方形。
    口径 B 已在 LUT 建表时完成，渲染时像素角距 a 直接查表。

    参数
    ----
    d_arcmin   : 月心到本影中心角距 (arcmin)，默认 22（食分≈1.1 全食）
    size       : 输出边长像素
    ssaa       : 超采样倍率（边缘抗锯齿），每边 ×ssaa，最后 box 下采样
    exposure   : 全局曝光标量 E；None 则按 target_srgb 反解（让趋白区映到该 sRGB）
    返回 (rgb8, info)
    """
    lut_kwargs = lut_kwargs or {}
    a_lut, XYZ_lum, Y_lum, meta = build_lut(**lut_kwargs)

    # 画幅：以月盘为中心略放余量
    half = R_MOON_ARCMIN + margin_arcmin
    cx_world = d_arcmin                      # 月盘中心 world x
    x0, x1 = cx_world - half, cx_world + half
    y0, y1 = -half, half

    # 超采样网格（world 角坐标 arcmin）
    S = size * ssaa
    xs = np.linspace(x0, x1, S)
    ys = np.linspace(y0, y1, S)
    X_world, Y_world = np.meshgrid(xs, ys)

    # 到本影中心(原点)的角距 a，到月心的距离 r_moon
    a = np.hypot(X_world, Y_world)
    r_moon = np.hypot(X_world - cx_world, Y_world)

    inside = r_moon <= R_MOON_ARCMIN

    # 查 LUT 得线性 XYZ（全网格查，月盘外稍后乘 0）
    XYZ_pix = lookup_xyz(a, a_lut, XYZ_lum, Y_lum)        # (S,S,3)

    # ---- 全局统一曝光标定 ----
    if exposure is None:
        # 摄影做法「为暗部曝光」：本影内动态范围极大（红核比绿松石暗~240×），
        # 单一曝光若按最亮端标定，红核会死黑。改按月盘上**较暗的红核侧**标定，
        # 让红核进入可见中调，绿松石/白靠 Reinhard 压高光自然收住，不过曝。
        # 全局统一曝光，绝不 per-pixel——保住「中心暗、边缘亮」的相对结构。
        a_near = max(d_arcmin - R_MOON_ARCMIN, a_lut[0])     # 月盘最靠本影中心一侧(红核)
        Y_dark = float(lookup_xyz(np.array([a_near]), a_lut, XYZ_lum, Y_lum)[0, 1])
        # 让红核侧映到中等偏暗 sRGB（保留它是暗部的观感，但能看见）
        t = np.clip(_srgb_inv_gamma(0.32), 1e-4, 0.999)
        exposure = t / (max(Y_dark, 1e-12) * (1.0 - t))

    # 在亮度上做 Reinhard（不动色相）：取 xyY，压 Y，再回 XYZ
    XYZ_disp = _tone_map_on_Y(XYZ_pix, exposure)

    # 线性 XYZ → 线性 sRGB → gamma → [0,1]
    rgb = _xyz_to_srgb_linear(XYZ_disp)
    rgb = _srgb_gamma(np.clip(rgb, 0.0, 1.0))

    # 月盘外置黑（天空）；月盘 mask 用覆盖率（超采样下采样自然实现 AA）
    rgb = rgb * inside[..., None]

    # 超采样下采样到目标尺寸（box filter）= 抗锯齿
    rgb8 = _box_downsample(rgb, ssaa)
    rgb8 = (np.clip(rgb8, 0, 1) * 255 + 0.5).astype(np.uint8)

    info = dict(a_lut=a_lut, XYZ_lum=XYZ_lum, Y_lum=Y_lum, meta=meta,
                exposure=float(exposure), d=d_arcmin,
                R_moon=R_MOON_ARCMIN, R_umbra=R_UMBRA_ARCMIN,
                extent=(x0, x1, y0, y1), cx_world=cx_world)
    return rgb8, info


# ============================================================
# 显示链辅助
# ============================================================
# sRGB↔linear（IEC 61966-2-1）
def _srgb_gamma(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(np.maximum(c, 0), 1 / 2.4) - 0.055)


def _srgb_inv_gamma(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))


# CIE XYZ (D65) → linear sRGB
_M_XYZ2RGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])


def _xyz_to_srgb_linear(XYZ):
    return XYZ @ _M_XYZ2RGB.T


def _tone_map_on_Y(XYZ, E, hi_max=0.90, ey_white=None):
    """全局曝光 + 对数高光肩部，只作用在亮度 Y 上（保色相/色度）。

    Y_disp = hi_max · log2(1 + E·Y) / log2(1 + ey_white)
    ey_white 是"映到 hi_max 的曝光亮度"白点。**必须固定**（不能用输入 max，
    否则退化成 per-frame/per-call 归一化——视频里会让每帧月亮一样亮）。
    缺省 ey_white=E·1（正常满亮度 Y=1 映到 hi_max）。暗部(本影深处)经对数曲线
    保持暗，不被拉亮。
    """
    X, Y, Z = XYZ[..., 0], XYZ[..., 1], XYZ[..., 2]
    Ys = np.maximum(Y, 1e-30)
    cx, cz = X / Ys, Z / Ys
    EY = E * Y
    if ey_white is None:
        ey_white = E * 1.0                       # 固定白点：正常满亮度 Y=1
    Yd = hi_max * np.log2(1.0 + EY) / np.log2(1.0 + max(ey_white, 1e-6))
    Yd = np.clip(Yd, 0.0, 1.0)
    return np.stack([cx * Yd, Yd, cz * Yd], axis=-1)


def _box_downsample(img, f):
    """整数倍 box 下采样 (S,S,C) -> (S/f, S/f, C)。"""
    if f == 1:
        return img
    S = img.shape[0]
    n = S // f
    img = img[:n * f, :n * f]
    return img.reshape(n, f, n, f, img.shape[2]).mean(axis=(1, 3))


# ============================================================
# 3. 自查 + 出图
# ============================================================
def self_check(info):
    """沿 O_umbra→O_moon 方向在月盘上采一条线，打印红核/绿松石/白边的角距范围与代表色 sRGB。"""
    a_lut, XYZ_lum, Y_lum = info["a_lut"], info["XYZ_lum"], info["Y_lum"]
    E = info["exposure"]
    d, R_moon = info["d"], info["R_moon"]

    print("\n=== 月盘渲染自查 ===")
    print(f"月盘半径 {R_moon:.1f}'  本影半径 {info['R_umbra']:.1f}'  月心偏移 d={d:.1f}'  全局曝光 E={E:.3g}")
    mag = (info["R_umbra"] + R_moon - d) / (2 * R_moon)
    print(f"食甚食分 mag={mag:.2f}（>1 为全食）")

    # 沿连线方向：world x 从 d-R_moon 到 d+R_moon，y=0；对应到本影中心角距 a=|x|
    xs = np.linspace(d - R_moon, d + R_moon, 400)
    a_line = np.abs(xs)
    XYZ_line = lookup_xyz(a_line, a_lut, XYZ_lum, Y_lum)
    XYZ_disp = _tone_map_on_Y(XYZ_line, E)
    rgb = _srgb_gamma(np.clip(_xyz_to_srgb_linear(XYZ_disp), 0, 1))
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    # 分类（与 pipeline.self_check 同口径：用 sRGB 通道关系）
    near_white = (R > 0.85) & (G > 0.85) & (B > 0.85)
    red_mask = (R > B + 0.06) & (R >= G) & (~near_white)
    teal_mask = (B >= R - 0.02) & (~near_white) & (G > 0.15) & (~red_mask)

    def rep(mask, name):
        if not mask.any():
            print(f"  {name}: 未出现")
            return
        amin, amax = a_line[mask].min(), a_line[mask].max()
        # 代表色取该段中位像素
        idx = np.where(mask)[0]
        mid = idx[len(idx) // 2]
        c = (rgb[mid] * 255).astype(int)
        print(f"  {name}: 角距 {amin:.1f}–{amax:.1f}'  代表色 sRGB=({c[0]},{c[1]},{c[2]})")

    rep(red_mask, "红核区(暖)")
    rep(teal_mask, "绿松石环(冷)")
    rep(near_white, "趋白/外缘")

    # 三色径向次序：沿月盘从近本影中心一侧(x小)到远侧(x大)，应 红→青→白
    print("\n  径向次序检查（x: 近中心→远边缘）:")
    seq = []
    for frac, label in [(0.10, "近中心"), (0.5, "盘中"), (0.92, "远边缘")]:
        i = int(frac * (len(xs) - 1))
        c = (rgb[i] * 255).astype(int)
        seq.append(f"{label} a={a_line[i]:.1f}' sRGB=({c[0]},{c[1]},{c[2]})")
    for s in seq:
        print("   ", s)

    order_ok = (red_mask.any() and teal_mask.any() and
                a_line[red_mask].mean() < a_line[teal_mask].mean())
    print(f"\n  [{'OK' if order_ok else 'XX'}] 红核角距均值 < 绿松石环角距均值（红在内、青在外）")
    return order_ok


def save_disk_png(rgb8, info, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(7, 7))
    ext = info["extent"]
    ax.imshow(rgb8, extent=[ext[0], ext[1], ext[2], ext[3]], origin="lower")
    ax.set_xlabel("角距 x (arcmin)，原点=本影中心")
    ax.set_ylabel("角距 y (arcmin)")
    ax.set_title(f"月全食月盘（真实几何，d={info['d']:.0f}', 食分≈{(info['R_umbra']+info['R_moon']-info['d'])/(2*info['R_moon']):.2f}）\n"
                 "月盘真大小真位置骑跨绿松石带：近本影中心侧红、中部绿松石、远侧白", fontsize=10)
    # 标本影边界圆弧
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(info["R_umbra"] * np.cos(th), info["R_umbra"] * np.sin(th),
            ls="--", color="gray", lw=0.8, alpha=0.6, label=f"本影边界 {info['R_umbra']:.0f}'")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"已存月盘图：{path}")


def save_disk_with_legend(rgb8, info, path):
    """月盘 + 旁边角距色带图例的对照图。"""
    a_lut, XYZ_lum, Y_lum = info["a_lut"], info["XYZ_lum"], info["Y_lum"]
    E = info["exposure"]
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1.4], height_ratios=[20, 1])

    ax = fig.add_subplot(gs[:, 0])
    ext = info["extent"]
    ax.imshow(rgb8, extent=[ext[0], ext[1], ext[2], ext[3]], origin="lower")
    ax.set_xlabel("角距 x (arcmin)，原点=本影中心")
    ax.set_ylabel("角距 y (arcmin)")
    ax.set_title(f"月全食月盘 2D 渲染（真实几何，月盘骑跨绿松石带，本影 {info['R_umbra']:.0f}'）",
                 fontsize=11)
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(info["R_umbra"] * np.cos(th), info["R_umbra"] * np.sin(th),
            ls="--", color="gray", lw=0.8, alpha=0.6)
    ax.set_aspect("equal")

    # 右上：角距色带图例（同一曝光链）
    a_axis = np.linspace(a_lut[0], a_lut[-1], 512)
    XYZ_band = lookup_xyz(a_axis, a_lut, XYZ_lum, Y_lum)
    rgb_band = _srgb_gamma(np.clip(_xyz_to_srgb_linear(_tone_map_on_Y(XYZ_band, E)), 0, 1))
    axb = fig.add_subplot(gs[0, 1])
    axb.imshow(rgb_band[None, :, :], aspect="auto", extent=[a_axis[0], a_axis[-1], 0, 1])
    axb.set_yticks([])
    axb.set_xlabel("距本影中心角距 (arcmin)\n小=红核  大=绿松石/白")
    axb.set_title("LUT 角距色带（同一曝光）\n暗红 → 绿松石 → 白", fontsize=9)

    # 右下：说明文字
    axt = fig.add_subplot(gs[1, 1])
    axt.axis("off")
    mag = (info["R_umbra"] + info["R_moon"] - info["d"]) / (2 * info["R_moon"])
    axt.text(0, 0.5, f"d={info['d']:.0f}'  食分≈{mag:.2f}  E={info['exposure']:.2g}\n"
                     "基于 L1 物理，色相微调中", fontsize=8, va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"已存对照图：{path}")


# ============================================================
# 4. 满画面全月盘渲染（红→绿松石→白完整展开）+ HDR
# ============================================================
def render_full_disk(size=1200, ssaa=2, a_inner=4.0, a_outer=None,
                     gradient_axis="x", target_srgb=0.92, lut_kwargs=None):
    """满画面月盘：月盘充满画幅，盘面颜色沿一个方向从 红核→绿松石→白 完整展开。

    与 render_disk 的区别：那个按真实月食几何（月盘小、偏心、只切本影边缘一道弧），
    红绿白展不开。这里取**展示/艺术取向**——把月盘视半径映射成完整角距区间
    [a_inner, a_outer]，让月盘一侧贴本影深处(红)、另一侧到本影边缘(白)，中间是绿松石。
    物理着色仍来自同一 LUT（角距→颜色），只是把"角距"沿盘面线性铺开，标题会注明。

    a_inner/a_outer: 月盘两端映射到的角距(arcmin)。a_outer 缺省 = 本影边界 R_umbra。
    gradient_axis: 'x' 渐变沿水平方向（左红右白），'radial' 同心（中心红外圈白）。
    返回 (rgb8_linear_HDR, rgb8_display, info) —— HDR 是线性高动态，display 是 tone-map 后 8bit。
    """
    lut_kwargs = lut_kwargs or {}
    if a_outer is None:
        a_outer = R_UMBRA_ARCMIN
    a_lut, XYZ_lum, Y_lum, meta = build_lut(**lut_kwargs)

    S = size * ssaa
    # 归一化盘面坐标 [-1,1]
    u = np.linspace(-1, 1, S)
    U, V = np.meshgrid(u, u)
    rr = np.hypot(U, V)
    inside = rr <= 1.0

    # 盘面位置 → 角距：沿选定方向把 [a_inner, a_outer] 铺开
    if gradient_axis == "x":
        # 左(U=-1)=a_inner(红核), 右(U=+1)=a_outer(白)
        frac = (U + 1.0) / 2.0
    else:  # radial：中心=a_inner，边缘=a_outer
        frac = np.clip(rr, 0, 1)
    a = a_inner + (a_outer - a_inner) * frac

    XYZ_pix = lookup_xyz(a, a_lut, XYZ_lum, Y_lum)        # 线性 XYZ (S,S,3)

    # 全局曝光：让最亮端(a_outer, 趋白)映到 target_srgb
    Y_bright = float(lookup_xyz(np.array([a_outer]), a_lut, XYZ_lum, Y_lum)[0, 1])
    t = np.clip(_srgb_inv_gamma(target_srgb), 1e-4, 0.999)
    E = t / (Y_bright * (1.0 - t))

    # HDR：线性 sRGB（曝光后，未 tone-map、未 gamma），保留高动态范围
    XYZ_exposed = XYZ_pix * E
    rgb_lin_hdr = np.clip(_xyz_to_srgb_linear(XYZ_exposed), 0, None)
    rgb_lin_hdr = rgb_lin_hdr * inside[..., None]

    # 显示：Reinhard tone-map on Y + gamma
    XYZ_disp = _tone_map_on_Y(XYZ_pix, E)
    rgb_disp = _srgb_gamma(np.clip(_xyz_to_srgb_linear(XYZ_disp), 0, 1))
    rgb_disp = rgb_disp * inside[..., None]

    hdr = _box_downsample(rgb_lin_hdr, ssaa)
    disp8 = (np.clip(_box_downsample(rgb_disp, ssaa), 0, 1) * 255 + 0.5).astype(np.uint8)

    info = dict(a_lut=a_lut, XYZ_lum=XYZ_lum, Y_lum=Y_lum, meta=meta, exposure=float(E),
                a_inner=a_inner, a_outer=a_outer, gradient_axis=gradient_axis)
    return hdr, disp8, info


def save_hdr(hdr_linear, path):
    """存线性 HDR。优先 .exr(OpenEXR)，无则存 .hdr(Radiance)，再不行存 16bit PNG 兜底。"""
    base = os.path.splitext(path)[0]
    try:
        import imageio.v2 as imageio
        imageio.imwrite(base + ".exr", hdr_linear.astype(np.float32))
        print(f"已存 HDR：{base}.exr")
        return base + ".exr"
    except Exception as e1:
        try:
            import imageio.v2 as imageio
            imageio.imwrite(base + ".hdr", hdr_linear.astype(np.float32))
            print(f"已存 HDR：{base}.hdr")
            return base + ".hdr"
        except Exception as e2:
            # 16bit PNG 兜底（归一到最大值，非真 HDR 但保留更多动态）
            m = max(hdr_linear.max(), 1e-6)
            png16 = (np.clip(hdr_linear / m, 0, 1) * 65535 + 0.5).astype(np.uint16)
            try:
                import imageio.v2 as imageio
                imageio.imwrite(base + "_16bit.png", png16)
                print(f"已存 16bit PNG(HDR兜底)：{base}_16bit.png（EXR/HDR 不可用: {e1}）")
                return base + "_16bit.png"
            except Exception as e3:
                print(f"HDR 存储失败: {e1} / {e2} / {e3}")
                return None


def save_full_disk_png(disp8, info, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(disp8, origin="lower")
    ax.axis("off")
    ax.set_title("月全食满画面月盘（展示取向：盘面铺开 红核→绿松石→白）\n"
                 f"着色为 L1 物理(角距→透射谱→sRGB)，角距区间 [{info['a_inner']:.0f}', {info['a_outer']:.0f}']",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"已存满画面月盘：{path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=float, default=22.0, help="月心到本影中心角距 arcmin")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--exposure", type=float, default=None)
    ap.add_argument("--full", action="store_true", help="满画面全月盘模式(红→绿松石→白完整展开)+HDR")
    ap.add_argument("--axis", default="x", choices=["x", "radial"], help="满画面渐变方向")
    args = ap.parse_args()

    if args.full:
        import time
        t0 = time.time()
        hdr, disp8, info = render_full_disk(size=args.size, ssaa=args.ssaa,
                                            gradient_axis=args.axis)
        print(f"满画面渲染 {args.size}x{args.size} (SSAA×{args.ssaa}) 用时 {time.time()-t0:.2f}s, 曝光 E={info['exposure']:.3g}")
        save_full_disk_png(disp8, info, os.path.join(OUT, f"moon_full_{args.axis}.png"))
        save_hdr(hdr, os.path.join(OUT, f"moon_full_{args.axis}_hdr"))
        sys.exit(0)

    import time
    t0 = time.time()
    rgb8, info = render_disk(d_arcmin=args.d, size=args.size, ssaa=args.ssaa,
                             exposure=args.exposure)
    dt = time.time() - t0
    print(f"渲染 {args.size}x{args.size} (SSAA×{args.ssaa}) 用时 {dt:.2f}s")

    ok = self_check(info)

    save_disk_png(rgb8, info, os.path.join(OUT, "moon_disk.png"))
    save_disk_with_legend(rgb8, info, os.path.join(OUT, "moon_disk_legend.png"))
    print(f"\n{'渲染自查通过' if ok else '径向次序未通过，检查口径/几何'}")
