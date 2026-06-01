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

# 位置: 绿松石带交界在横向 2/3 处。圆盘版最蓝在 40', 点源版在 ~52'。
# 用 d 让"该步的绿松石带角距"落在月盘横向 2/3。月盘横向范围 [d-Rm, d+Rm], 2/3 处 = d-Rm + (4/3)Rm = d + Rm/3。
# 要 R/B 谷在 2/3 处: d + Rm/3 = a_blue → d = a_blue - Rm/3。
# 但前几步(点源)和后几步(圆盘)蓝带位置不同。为视频连贯, 统一用一个 d。
# 折中: 用点源蓝带~50'(步4高潮所在), d = 50 - Rm/3 ≈ 45。这样步4青带在2/3, 步5圆盘青带会左移些。
D_ABLATION = 45.0


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
    """圆盘 a→XYZ LUT(金标准 brute_trace)。solar_mode 经 brute_trace 内部(默认real)。"""
    import importlib
    res = bt.brute_trace(n_h=200000, n_xi=257, bin_width=0.08,
                         a_grid_lo=18, a_grid_hi=72, n_disp=n_disp)
    a = res["a"]; XYZ = res["XYZ"].copy(); Y = res["Y"]
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
STEPS = [
    dict(key="1_disk",    title="① 土圆盘 + 月面纹理",
         note="最土的起点：一个有月海纹理的灰白月亮（无任何食光物理）",
         mode="flat_tex"),
    dict(key="2_geom",    title="② + 几何遮挡（无折射、黑白）",
         note="日食几何：若没有大气折射，地球完全挡住太阳，本影内该全黑、外全白（硬边）",
         mode="geom"),
    dict(key="3_rayleigh", title="③ + 瑞利散射",
         note="本影内不再全黑——瑞利散射 ∝λ⁻⁴ 把蓝光散尽，只剩红光：血月",
         mode="point", kw=dict(ozone=False)),
    dict(key="4_ozone",   title="④ + 臭氧 Chappuis 吸收",
         note="红的内沿冒出一条青带！臭氧吃掉橙红光（500-650nm）——这就是绿松石带",
         mode="point", kw=dict()),
    dict(key="5_disk",    title="⑤ + 太阳圆盘（有限元）",
         note="太阳不是点：16′圆盘比青带还宽5倍，把浓青糊成浅青软边、并左移——真实没那么浓",
         mode="disk", n_disp=1),
    dict(key="6_full",    title="⑥ + 实测太阳谱 + 折射色散",
         note="最后两个二阶修正：实测太阳谱微调色相、大气色散让边缘更柔——最终真实效果",
         mode="disk", n_disp=16),
]


def render_step(step, size=1200, ssaa=2, enhance=True, d=D_ABLATION):
    """渲染一步。enhance=True 夸张增强(对比图/视频), False 真实版(步6结尾)。"""
    S = size * ssaa
    half = Rm + 3.0
    xs = np.linspace(d - half, d + half, S)
    ys = np.linspace(-half, half, S)
    Xw, Yw = np.meshgrid(xs, ys)
    a = np.hypot(Xw, Yw)
    U = (Xw - d) / Rm; V = Yw / Rm
    inside = np.hypot(U, V) <= 1.0
    z = np.sqrt(np.clip(1 - U*U - V*V, 0, 1))

    # 月面纹理(正交投影)
    alb_tex, chroma_tex = rtx.load_albedo_texture()
    alb, mu, (ri, ci) = rtx.sample_albedo_orthographic(alb_tex, U, V, inside)
    limb = np.power(np.clip(mu, 0, 1), 0.5)

    import geometry as g
    mode = step["mode"]
    if mode == "flat_tex":
        # 土圆盘+纹理: 纯反照率×白光(无食光), 满月态
        XYZ_phys = np.ones((S, S, 3)) * np.array([0.95, 1.0, 1.05])  # 接近白(中性月光)
    elif mode == "geom":
        # 纯几何遮挡(无折射): 本影内(a<本影半径)=黑(光线被地球直挡), 外=白。硬边。
        # 这是"如果没有大气折射, 日食月亮该全黑"的教学态——揭示折射是月食不全黑的原因。
        Rumbra = g.umbra_radius_arcmin()
        XYZ_phys = np.where((a <= Rumbra)[..., None], 0.0, np.array([0.95, 1.0, 1.05]))
    elif mode == "point":
        lut = _build_lut_pointsource(**step.get("kw", {}))
        XYZ_phys = _shade_lut(a, lut)
    elif mode == "disk":
        lut = _build_lut_disk(n_disp=step.get("n_disp", 1))
        XYZ_phys = _shade_lut(a, lut)

    XYZ_scene = XYZ_phys * (alb / 0.12 * limb)[..., None]

    # tone map
    Yp = np.maximum(XYZ_scene[..., 1], 1e-30); chroma = XYZ_scene / Yp[..., None]
    if enhance:
        dg, sat, pct, tgt = 0.5, 1.6, 90, 0.88   # 夸张: 强压缩+提饱和
    else:
        dg, sat, pct, tgt = 0.6, 1.2, 95, 0.90   # 真实: 温和
    Yc = np.power(Yp, dg); XYZc = chroma * Yc[..., None]
    Yb = np.percentile(XYZc[..., 1][inside], pct)
    t = R._srgb_inv_gamma(tgt); E = t / (max(Yb, 1e-12) * (1 - t))
    rgb = np.clip(R._xyz_to_srgb_linear(R._tone_map_on_Y(XYZc, E)), 0, None)
    # 月面色偏(增写实)
    ctex = chroma_tex[ri, ci]; ctex = 1.0 + 0.5 * (ctex - 1.0); rgb = np.clip(rgb * ctex, 0, None)
    # 饱和
    luma = (0.2126*rgb[..., 0]+0.7152*rgb[..., 1]+0.0722*rgb[..., 2])[..., None]
    rgb = np.clip(luma + (rgb - luma) * sat, 0, None)
    rgb = R._srgb_gamma(np.clip(rgb, 0, 1)) * inside[..., None]
    rgb8 = R._box_downsample(rgb, ssaa)
    return (np.clip(rgb8, 0, 1) * 255 + 0.5).astype(np.uint8)


def add_annotations(rgb8, title, note, enhanced=True):
    """加版式: 左上=改动标题, 左下=小字注释。"""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(rgb8); d = ImageDraw.Draw(im); W, H = im.size
    def font(sz):
        for fp in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                   "/System/Library/Fonts/STHeiti Medium.ttc"]:
            try: return ImageFont.truetype(fp, sz)
            except Exception: pass
        return ImageFont.load_default()
    # 左上标题(大)
    d.text((28, 24), title, fill=(255, 255, 255), font=font(int(H*0.040)))
    # 左下注释(小字)
    note_y = H - int(H*0.075)
    d.text((28, note_y), note, fill=(200, 210, 220), font=font(int(H*0.024)))
    if enhanced:
        # 增强标注(右下角小字)
        en = "图像经对比度夸张增强，以更鲜明显示该因素的效果"
        ft = font(int(H*0.020))
        bb = d.textbbox((0,0), en, font=ft)
        d.text((W - (bb[2]-bb[0]) - 28, H - int(H*0.040)), en, fill=(150, 160, 175), font=ft)
    return np.asarray(im)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1200)
    ap.add_argument("--ssaa", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(os.path.join(OUT, "ablation"), exist_ok=True)
    from PIL import Image
    for i, step in enumerate(STEPS):
        enh = render_step(step, size=args.size, ssaa=args.ssaa, enhance=True)
        enh = add_annotations(enh, step["title"], step["note"], enhanced=True)
        p = os.path.join(OUT, "ablation", f"step_{step['key']}.png")
        Image.fromarray(enh).save(p)
        print(f"  saved {p}")
    # 步6真实版(视频结尾)
    real = render_step(STEPS[-1], size=args.size, ssaa=args.ssaa, enhance=False)
    real = add_annotations(real, "最终：真实物理效果", "不经增强——这就是太阳圆盘下绿松石带的真实样子：浅青、靠内、柔和", enhanced=False)
    Image.fromarray(real).save(os.path.join(OUT, "ablation", "step_7_real.png"))
    print("  saved step_7_real.png (真实版)")
    print("ablation 6步+真实版 渲染完成")
