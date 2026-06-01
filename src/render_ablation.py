"""Ablation study 逐层渲染：6 步从最土到真实，每步加一个物理因素。

叙事张力排序(见 docs/ablation_plan.md):
  1 土圆盘+纹理  2 几何遮挡(黑白)  3 瑞利(血月红)  4 臭氧(绿松石带)
  5 太阳圆盘(柔化)  6 太阳谱+色散(真实)

绿松石带交界放横向 2/3 处(不过中心)。两套输出:
  - 夸张增强版(对比图+翻页视频): 强动态范围压缩+提饱和, 配版式注释。
  - 真实版(步6不增强, 视频结尾用)。

复用 brute_ray_trace(圆盘) / radiative_transfer(开关) / render_textured(纹理几何)。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import brute_ray_trace as bt
import radiative_transfer as rt
import color as col
import solar
import render as R
import render_textured as rtx

OUT = R.OUT
Rm = R.R_MOON_ARCMIN

# 位置 normalize: 绿松石带固定在月盘横向 2/3 处(d + Rm/3 = a_blue → d = a_blue - Rm/3)。
# 加太阳圆盘后青带从点源52.6'移到圆盘40.4', 若固定d青带会偏移、观众难懂。所以每步单独调d,
# 让该步青带都落2/3, 差别只剩颜色本身。无青带步(1-3)用步4(臭氧点源)的d对齐, 月盘几何位置一致。
# (各 a_blue 由物理算得: 点源臭氧52.6', 圆盘40.4'; d = a_blue - Rm/3, Rm/3≈5.2')
D_POINT_BLUE = 47.4    # 步1-4: 点源青带@52.6→2/3
D_DISK_BLUE = 35.2     # 步5-6: 圆盘青带@40.4→2/3


def _build_lut_pointsource(**kw):
    """点源 a→XYZ LUT(各擦边高度的食光色 + a_signed 落点)。kw 传 emergent_spectrum 开关。"""
    lam = np.linspace(380, 780, 401)
    white = col.spectrum_to_XYZ(lam, solar.solar_spectrum(lam)); k = 1.0 / white[1]
    h = np.linspace(0, 80, 4000)
    I = rt.emergent_spectrum(h, lam, **kw)
    XYZ = np.array([col.spectrum_to_XYZ(lam, I[i]) for i in range(len(h))]) * k
    a = bt.a_signed_arcmin(h)
    # 单调段(从极小值起)
    i_min = int(np.argmin(a)); sl = slice(i_min, None)
    a_mono = a[sl]; XYZ_mono = XYZ[sl]
    # 趋白外缘归一 + 边界外clamp白
    yref = np.percentile(XYZ_mono[:, 1], 99); XYZ_mono = XYZ_mono / max(yref, 1e-9)
    i_peak = int(np.argmax(XYZ_mono[:, 1]))
    XYZ_mono = XYZ_mono.copy(); XYZ_mono[i_peak:] = XYZ_mono[i_peak]
    return dict(a=a_mono, XYZ=XYZ_mono)


def _build_lut_disk(n_disp=1, solar_mode="real"):
    """圆盘 a→XYZ LUT(金标准 brute_trace)。分箱有蒙特卡洛统计噪声→高分辨率渲染可见banding,
    用高斯平滑(sigma=2bins≈0.16', 远小于绿松石带4'尺度)消噪不损物理(最蓝R/B不变)。"""
    from scipy.ndimage import gaussian_filter1d
    res = bt.brute_trace(n_h=200000, n_xi=257, bin_width=0.08,
                         a_grid_lo=18, a_grid_hi=72, n_disp=n_disp)
    a = res["a"]; XYZ = res["XYZ"].copy(); Y = res["Y"]
    for c in range(3):
        XYZ[:, c] = gaussian_filter1d(XYZ[:, c], sigma=2.0)   # 消分箱噪声(banding)
    yref = np.percentile(XYZ[Y > 0, 1], 99); XYZ = XYZ / max(yref, 1e-9)
    i_peak = int(np.argmax(XYZ[:, 1])); XYZ[i_peak:] = XYZ[i_peak]
    return dict(a=a, XYZ=XYZ)


def _shade_lut(a_pixel, lut):
    a = np.asarray(a_pixel, float)
    out = np.empty(a.shape + (3,))
    for c in range(3):
        out[..., c] = np.interp(a, lut["a"], lut["XYZ"][:, c],
                                left=lut["XYZ"][0, c], right=lut["XYZ"][-1, c])
    return out


# ── 6 步定义 ────────────────────────────────────────────────────────────
# 每步: (key, 标题, 注释, LUT构造, 是否带纹理, 是否夸张)
# d: normalize 青带到 2/3。步1-4(点源几何)用 D_POINT_BLUE, 步5-6(圆盘)用 D_DISK_BLUE。
STEPS = [
    dict(key="1_disk",    title="① 土圆盘 + 月面纹理",
         note="最土的起点：一个有月海纹理的灰白月亮（无任何食光物理）",
         mode="flat_tex", d=D_POINT_BLUE),
    dict(key="2_geom",    title="② + 几何遮挡（无折射、黑白）",
         note="日食几何：若没有大气折射，地球完全挡住太阳，本影内该全黑、外全白（硬边）",
         mode="geom", d=D_POINT_BLUE),
    dict(key="3_rayleigh", title="③ + 瑞利散射",
         note="本影内不再全黑——瑞利散射 ∝λ⁻⁴ 把蓝光散尽，只剩红光：血月",
         mode="point", kw=dict(ozone=False), d=D_POINT_BLUE),
    dict(key="4_ozone",   title="④ + 臭氧 Chappuis 吸收",
         note="红的内沿冒出一条青带！臭氧吃掉橙红光（500-650nm）——这就是绿松石带",
         mode="point", kw=dict(), d=D_POINT_BLUE),
    dict(key="5_disk",    title="⑤ + 太阳圆盘（有限元）",
         note="太阳不是点：16′圆盘比青带还宽5倍，把浓青糊成浅青软边——真实没那么浓",
         mode="disk", n_disp=1, d=D_DISK_BLUE),
    dict(key="6_full",    title="⑥ + 实测太阳谱 + 折射色散",
         note="最后两个二阶修正：实测太阳谱微调色相、大气色散让边缘更柔——最终真实效果",
         mode="disk", n_disp=16, d=D_DISK_BLUE),
]


# HDR nits tone mapping(与视频月面 panel 同一套): nits = black + exp·Y·white。
# 输出线性 nits 值(16bit TIFF), 剪辑软件里自己微调色相/对比。不做夸张增强/饱和/色偏。
MOON_HDR_EXP, MOON_HDR_BLACK, MOON_HDR_WHITE = 1.0, 0.1, 200.0   # 同 build_video


def render_step(step, size=1200, ssaa=2):
    """渲染一步 → 线性 RGB(HDR nits 映射, 月面 panel 同套 tone mapping)。
    返回 (rgb_lin_nits float32 (size,size,3)) 供存 16bit TIFF。绿松石带已 normalize 到 2/3。
    """
    d = step.get("d", D_POINT_BLUE)
    S = size * ssaa
    half = Rm + 3.0
    xs = np.linspace(d - half, d + half, S)
    ys = np.linspace(-half, half, S)
    Xw, Yw = np.meshgrid(xs, ys)
    a = np.hypot(Xw, Yw)
    U = (Xw - d) / Rm; V = Yw / Rm
    inside = np.hypot(U, V) <= 1.0

    # 月面纹理(正交投影)
    alb_tex, chroma_tex = rtx.load_albedo_texture()
    alb, mu, (ri, ci) = rtx.sample_albedo_orthographic(alb_tex, U, V, inside)
    limb = np.power(np.clip(mu, 0, 1), 0.5)

    import geometry as g
    mode = step["mode"]
    if mode == "flat_tex":
        XYZ_phys = np.ones((S, S, 3)) * np.array([0.95, 1.0, 1.05])
    elif mode == "geom":
        # 本影内: 不是纯黑, 留极暗月面纹理(月亮还在、只是没被照亮)——作为连续锚点贯穿6步,
        # 也符合真实(地球反照/星光让无折射全食月仍隐约可见)。本影外=直射白。
        Rumbra = g.umbra_radius_arcmin()
        umbra = (a <= Rumbra)[..., None]
        white = np.array([0.95, 1.0, 1.05])
        dark = white * 0.04          # 本影内极暗(满量程~4%, 纹理由后面 alb 调制显出来)
        XYZ_phys = np.where(umbra, dark, white)
    elif mode == "point":
        lut = _build_lut_pointsource(**step.get("kw", {}))
        XYZ_phys = _shade_lut(a, lut)
    elif mode == "disk":
        lut = _build_lut_disk(n_disp=step.get("n_disp", 1))
        XYZ_phys = _shade_lut(a, lut)

    # 反照率 × 食光 × limb → 线性场景 XYZ
    XYZ_scene = XYZ_phys * (alb / 0.12 * limb)[..., None]
    rgb_lin = np.clip(R._xyz_to_srgb_linear(XYZ_scene), 0, None) * inside[..., None]

    # HDR nits tone mapping(月面 panel 同套): nits = black + exp·Y·white(线性, 不压gamma)
    Y = np.maximum(0.2126*rgb_lin[..., 0]+0.7152*rgb_lin[..., 1]+0.0722*rgb_lin[..., 2], 1e-12)
    nits_Y = MOON_HDR_BLACK + MOON_HDR_EXP * Y * MOON_HDR_WHITE
    rgb_nits = rgb_lin * (nits_Y / Y)[..., None] * inside[..., None]

    return R._box_downsample(rgb_nits, ssaa).astype(np.float32)


# 全局定标(所有6张共用→tone mapping globally consistent, 跨图可比)。
# 全局峰值~236nit(满月白盘), 定标到满量程让最亮像素撑满16bit、不溢出。
ABLATION_NITS_FULL = 240.0    # 6张共用: 240nit → 65535(满量程)


def save_tiff16(rgb_nits, path):
    """存 16-bit 线性 TIFF。全局固定定标(ABLATION_NITS_FULL→满量程), 6张可比, linear。
    最亮像素(满月白~236nit)接近撑满65535; 修图软件里线性、跨图对比有意义。
    """
    import tifffile
    arr16 = np.clip(rgb_nits / ABLATION_NITS_FULL * 65535.0, 0, 65535).astype(np.uint16)
    tifffile.imwrite(path, arr16)


def save_preview_png(rgb_nits, path):
    """存 8bit PNG 预览(同全局定标+gamma压到可视, 仅供肉眼检查, 不是交付物)。"""
    from PIL import Image
    prev = np.clip(rgb_nits / ABLATION_NITS_FULL, 0, 1) ** (1/2.2)
    Image.fromarray((prev * 255).astype(np.uint8)).save(path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1400)
    ap.add_argument("--ssaa", type=int, default=2)
    args = ap.parse_args()
    abldir = os.path.join(OUT, "ablation")
    os.makedirs(abldir, exist_ok=True)
    for step in STEPS:
        rgb = render_step(step, size=args.size, ssaa=args.ssaa)
        save_tiff16(rgb, os.path.join(abldir, f"step_{step['key']}.tif"))
        save_preview_png(rgb, os.path.join(abldir, f"step_{step['key']}_preview.png"))
        print(f"  saved step_{step['key']}.tif (16bit线性nits) + preview")
    print("ablation 6步 16bit线性TIFF渲染完成(HDR nits tone map, 绿松石带normalize到2/3)")
    print(f"交付: {abldir}/step_*.tif — 剪辑软件里调色相/对比(全局定标 {ABLATION_NITS_FULL}nit→满量程, 6张可比)")
