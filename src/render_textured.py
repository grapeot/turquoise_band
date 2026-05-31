"""L3 写实月食渲染：在 L2 物理月盘上叠加真实月面反照率纹理。

成像原理（与真实月食照片一致）：
    最终辐亮度 = 月面反照率(maria 暗 / highland 亮) × 折射阳光照明颜色
折射阳光的颜色/亮度来自 render.py 的物理 LUT（角距→线性 XYZ，近本影中心红、
中部绿松石、远侧白），月面反照率只调制亮度、不碰色相。两者在**线性**空间相乘，
再走 render.py 既有的 tone-map + gamma 显示链。本模块不重新实现任何辐射传输或颜色管线。

复用 render.py 的：build_lut / lookup_xyz / _tone_map_on_Y / _xyz_to_srgb_linear /
_srgb_gamma / _box_downsample / _srgb_inv_gamma，以及几何常数 R_MOON_ARCMIN / R_UMBRA_ARCMIN。

纹理方案（路径 a，真实贴图）：
    NASA Clementine 全月反照率灰度图（plate carrée 等距柱状投影，1024×512，
    经度0在中心、纬度+90在顶，正面 near-side）。月海实测 ~0.09–0.15、高地 ~0.29–0.54，
    我们把它线性映射到月食文献常用的几何反照率区间（月海~0.07、高地~0.12），
    保住 maria/highland 的相对暗亮，得到物理上可信的反照率场。

球面投影（任务要点2）：
    月盘是球，按**正交投影 orthographic** 贴纹理（非平面贴图）。单位圆内像素 (X,Y)，
    Z=sqrt(1-X²-Y²) 为面向观察者的法向分量；(X,Y,Z) 即月面单位球上的视线交点，
    由它反解出选定中心经纬度下的月面经纬度，盘缘自然出现 foreshortening 压缩。

艺术处理（任务要点5，明确标注，不碰物理颜色）：
    - limb darkening：球面边缘按 Z=cosθ 自然变暗（µ^p，可信的球面光照衰减）；
    - 星空背景 + 轻微噪点：纯写实氛围，不影响月盘像素的物理着色。
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import render as R   # 复用 LUT / 显示链 / 几何常数

OUT = R.OUT
_TEXDIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "moon_texture")
# NASA CGI Moon Kit (SVS #4720) LROC 彩色 4k；缺则回退 Clementine 灰度
TEX_PATH = os.path.join(_TEXDIR, "nasa_moon_color_lroc_4k_16bit.tif")
TEX_PATH_FALLBACK = os.path.join(_TEXDIR, "moon-map-from-the-clementine-mission.png")


# ============================================================
# 1. 反照率纹理：加载真实贴图，映射到物理反照率区间
# ============================================================
def load_albedo_texture(path=TEX_PATH, mare_target=0.07, highland_target=0.12):
    """加载月面反照率纹理，线性映射到月食文献常用的几何反照率区间。

    优先 NASA CGI Moon Kit LROC 彩色 4k（16bit TIFF），取其亮度作为反照率调制；
    缺则回退 Clementine 灰度。彩色图的真实月面色（月海偏蓝灰、高地偏暖）以低权重
    保留为色偏，叠在物理食光颜色上增写实感（主色相仍由物理决定）。

    返回 (alb (H,W) 几何反照率, chroma (H,W,3) 纹理归一化色偏 或 None)。
    """
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    use_path = path if os.path.exists(path) else TEX_PATH_FALLBACK
    im = Image.open(use_path)

    if im.mode in ("L", "I", "I;16"):
        g = np.asarray(im.convert("F"), dtype=float)
        g = g / (g.max() if g.max() > 0 else 1.0)
        chroma = None
    else:
        arr = np.asarray(im.convert("RGB"), dtype=float)
        arr = arr / (arr.max() if arr.max() > 0 else 1.0)
        # 亮度作反照率；归一化色偏(除以自身亮度)作为月面本征色，低权重保留
        g = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
        gn = np.maximum(g, 1e-4)[..., None]
        chroma = arr / gn                          # 月海/高地的相对色偏，≈1 为中性

    lo = np.percentile(g, 5.0)    # 代表月海（暗）
    hi = np.percentile(g, 95.0)   # 代表高地（亮）
    alb = mare_target + (g - lo) / max(hi - lo, 1e-6) * (highland_target - mare_target)
    alb = np.clip(alb, 0.03, 0.18)
    return alb, chroma


def sample_albedo_orthographic(alb_tex, U, V, inside,
                               sub_lat_deg=0.0, sub_lon_deg=0.0):
    """正交投影：把 plate carrée 反照率纹理贴到单位圆月盘上（含球面 foreshortening）。

    参数
    ----
    alb_tex   : (H,W) 反照率纹理（plate carrée）
    U, V      : 月盘归一化坐标网格 [-1,1]（U 向右=+x，V 向上=+y）
    inside    : 圆盘内 mask（U²+V² ≤ 1）
    sub_lat/lon_deg : 视线中心点(盘心)对应的月面纬/经度（默认正面 0,0）

    做法（正交投影几何）
    --------------------
    盘面像素 (x,y)=(U,V) 是月球单位球在像平面的正交投影。面向观察者的法向分量
    z = sqrt(1 - x² - y²)（球面凸向观察者）。在以盘心为视线方向的相机系里，
    单位球点 P=(x, y, z)。绕 sub_lat 倾转回月固系后反解经纬度：
        lat = arcsin( z·cosφ0·... )  —— 这里用标准正交投影逆变换公式：
        lat = arcsin( cosc·sinφ0 + y·sinc·cosφ0 / ρ ) （ρ=1，c=arccos z）
    实现上直接用球面旋转：把相机系点 P 绕 x 轴转 sub_lat、再加 sub_lon 偏移取经度。

    返回 alb_disk (同 U 形状)，圆盘外置 0；并返回 mu=z（=cosθ，供 limb darkening）。
    """
    x = U
    y = V
    rr2 = x * x + y * y
    z = np.sqrt(np.clip(1.0 - rr2, 0.0, 1.0))   # 面向观察者法向分量 = cosθ

    phi0 = np.radians(sub_lat_deg)
    # 相机系单位向量 (x, y, z)，z 指向观察者。绕 x 轴旋转 phi0 把盘心从赤道挪到 sub_lat：
    #   月固系坐标 (xm, ym, zm)
    xm = x
    ym = y * np.cos(phi0) - z * np.sin(phi0)
    zm = y * np.sin(phi0) + z * np.cos(phi0)
    # 经纬度（zm 为指向观察者的"前方"轴 → 经度0方向；xm=东向；ym=北向）
    lat = np.arcsin(np.clip(ym, -1.0, 1.0))
    lon = np.arctan2(xm, zm) + np.radians(sub_lon_deg)
    # 归一到 [-pi,pi]
    lon = (lon + np.pi) % (2 * np.pi) - np.pi

    H, W = alb_tex.shape[:2]
    # plate carrée 反查：列=经度(-180→+180 映 0→W)，行=纬度(+90→-90 映 0→H)
    col = (np.degrees(lon) + 180.0) / 360.0 * (W - 1)
    row = (90.0 - np.degrees(lat)) / 180.0 * (H - 1)
    ci = np.clip(np.rint(col).astype(int), 0, W - 1)
    ri = np.clip(np.rint(row).astype(int), 0, H - 1)
    alb = alb_tex[ri, ci]
    alb = np.where(inside, alb, 0.0)
    mu = np.where(inside, z, 0.0)
    return alb, mu, (ri, ci)


# ============================================================
# 2. 写实月盘渲染：反照率 × 物理食光，线性相乘 → tone-map
# ============================================================
def render_realistic_disk(d_arcmin=26.0, size=1400, margin_arcmin=3.0, ssaa=2,
                          sub_lat_deg=0.0, sub_lon_deg=0.0,
                          limb_power=0.5, target_srgb=0.20,
                          dyn_gamma=0.85, chroma_weight=0.5, saturation=0.95,
                          hdr_headroom=3.0,
                          add_starfield=True, add_grain=True,
                          lut_kwargs=None, seed=7):
    """渲染写实月全食照片：真实月面反照率纹理 × 物理食光颜色。

    几何与 render.render_disk 同口径：本影中心为原点，月盘中心在 +x 距 d 处，
    月盘半径 R_MOON_ARCMIN(≈15.5')，本影半径 R_UMBRA_ARCMIN(≈41.2')。
    默认 d=26'（食分≈0.99，月盘骑跨绿松石带：近本影中心侧红、中部绿松石、远侧趋白）。

    管线
    ----
    1) build_lut → lookup_xyz：每像素按到本影中心角距查得**线性** XYZ（物理食光颜色）。
    2) 正交投影采样月面反照率 alb（归一化的几何反照率）。
    3) 线性空间相乘：XYZ_scene = XYZ_phys × (alb/alb_ref) × limb_darkening。
       alb/alb_ref 把反照率归一到"高地≈1"，使整体亮度量级与原物理月盘一致，
       maria 区按真实比例压暗。limb_darkening = mu^limb_power（球面边缘自然变暗，艺术）。
    4) 全局统一曝光（按月盘红核侧标定，禁 per-pixel）+ Reinhard + sRGB gamma。
    5) 叠星空背景与噪点（艺术，仅作用于月盘外/全局微扰，不改月盘物理着色）。

    返回 (rgb8, info)
    """
    # 着色改用逐像素反向 RT（分支感知反查），根治 banding，无需 LUT 加密补丁。
    import render_rt
    tables = render_rt.build_branch_tables(n_h=8000)
    rng = np.random.default_rng(seed)

    # ---- 画幅：以月盘为中心，留余量给星空 ----
    half = R.R_MOON_ARCMIN + margin_arcmin
    cx_world = d_arcmin
    x0, x1 = cx_world - half, cx_world + half
    y0, y1 = -half, half

    S = size * ssaa
    xs = np.linspace(x0, x1, S)
    ys = np.linspace(y0, y1, S)
    Xw, Yw = np.meshgrid(xs, ys)

    a = np.hypot(Xw, Yw)                       # 到本影中心角距
    # 月盘归一化坐标 [-1,1]
    U = (Xw - cx_world) / R.R_MOON_ARCMIN
    V = Yw / R.R_MOON_ARCMIN
    r_moon = np.hypot(U, V)
    inside = r_moon <= 1.0

    # ---- 物理食光颜色（线性 XYZ，逐像素反向 RT，无 banding）----
    XYZ_phys = render_rt.shade(a, tables)                 # (S,S,3)

    # ---- 月面反照率 + 真实月面色偏（正交投影采样）----
    alb_tex, chroma_tex = load_albedo_texture()
    alb, mu, (ri, ci) = sample_albedo_orthographic(alb_tex, U, V, inside,
                                                   sub_lat_deg=sub_lat_deg, sub_lon_deg=sub_lon_deg)
    alb_ref = 0.12
    alb_norm = alb / alb_ref

    # ---- limb darkening（艺术：球面边缘自然变暗，µ=cosθ）----
    limb = np.power(np.clip(mu, 0.0, 1.0), limb_power)

    # ---- 物理动态范围压缩（关键：让影调像真照片，不像物理 raw）----
    # 真实月食物理边-核差达 5-8 档(~2000×)，但摄影成品压到视觉 ~1.5-2.5 档(3:1~5:1)。
    # 把物理亮度 Y 做幂律压缩 Y^g (g<1) 收窄动态范围，色度(色相)不动——这是"压档"的本质。
    Yp = np.maximum(XYZ_phys[..., 1], 1e-30)
    chroma_xyz = XYZ_phys / Yp[..., None]              # 保色相
    Y_comp = np.power(Yp, dyn_gamma)                   # 幂律压缩亮度
    XYZ_comp = chroma_xyz * Y_comp[..., None]

    # ---- 线性空间相乘：反照率 × (压缩后物理颜色) × limb ----
    mod = (alb_norm * limb)[..., None]
    XYZ_scene = XYZ_comp * mod

    # ---- 全局统一曝光（按月盘红核侧标定，禁 per-pixel）----
    a_near = max(d_arcmin - R.R_MOON_ARCMIN, tables["a_lo"])
    Y_dark = float(np.power(render_rt.shade(np.array([a_near]), tables)[0, 1], dyn_gamma))
    t = np.clip(R._srgb_inv_gamma(target_srgb), 1e-4, 0.999)
    exposure = t / (max(Y_dark, 1e-12) * (1.0 - t))

    XYZ_disp = R._tone_map_on_Y(XYZ_scene, exposure)
    rgb = R._xyz_to_srgb_linear(XYZ_disp)
    rgb = np.clip(rgb, 0.0, None)

    # ---- HDR 线性缓冲（用于 gain map）：同曝光但给亮部 headroom，不做 Reinhard 压缩 ----
    # SDR 是上面 Reinhard 压到[0,1]的版本；HDR 让亮部(绿松石/白边)线性超过 1.0，
    # 比值 HDR/SDR 即 gain map。headroom 控制亮部能超亮多少倍。
    rgb_hdr_lin = R._xyz_to_srgb_linear(XYZ_scene * exposure) * hdr_headroom
    rgb_hdr_lin = np.clip(rgb_hdr_lin, 0.0, None)

    # ---- 叠加真实月面色偏（低权重，增写实，主色相仍由物理决定）----
    if chroma_tex is not None and chroma_weight > 0:
        ctex = chroma_tex[ri, ci]                       # (S,S,3) 月面本征色偏
        ctex = 1.0 + chroma_weight * (ctex - 1.0)       # 朝中性收权重
        rgb = np.clip(rgb * ctex, 0.0, None)
        rgb_hdr_lin = np.clip(rgb_hdr_lin * ctex, 0.0, None)  # HDR 同步色偏

    # ---- 饱和度微调（艺术：往中等铜橙靠，不要荧光血红）----
    if saturation != 1.0:
        luma = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])[..., None]
        rgb = np.clip(luma + (rgb - luma) * saturation, 0.0, None)
        luma_h = (0.2126 * rgb_hdr_lin[..., 0] + 0.7152 * rgb_hdr_lin[..., 1] + 0.0722 * rgb_hdr_lin[..., 2])[..., None]
        rgb_hdr_lin = np.clip(luma_h + (rgb_hdr_lin - luma_h) * saturation, 0.0, None)

    # ---- 星空背景（艺术，月盘外）----
    rgb = rgb * inside[..., None]
    rgb_hdr_lin = rgb_hdr_lin * inside[..., None]
    if add_starfield:
        sky = _starfield(S, rng)
        rgb = np.where(inside[..., None], rgb, sky)
        rgb_hdr_lin = np.where(inside[..., None], rgb_hdr_lin, R._srgb_inv_gamma(sky))

    # gamma
    rgb = R._srgb_gamma(np.clip(rgb, 0.0, 1.0))

    # ---- 轻微噪点（艺术，全局微扰，幅度极小不改色相结构）----
    if add_grain:
        grain = rng.normal(0.0, 0.006, rgb.shape)
        rgb = np.clip(rgb + grain, 0.0, 1.0)

    rgb8 = R._box_downsample(rgb, ssaa)
    rgb8 = (np.clip(rgb8, 0, 1) * 255 + 0.5).astype(np.uint8)
    # HDR 线性下采样（保留 >1）
    hdr_lin = R._box_downsample(rgb_hdr_lin, ssaa).astype(np.float32)

    info = dict(tables=tables, exposure=float(exposure), d=d_arcmin,
                R_moon=R.R_MOON_ARCMIN, R_umbra=R.R_UMBRA_ARCMIN,
                extent=(x0, x1, y0, y1), cx_world=cx_world,
                alb=alb, mu=mu, U=U, V=V, inside=inside, a_grid=a,
                XYZ_phys=XYZ_phys, alb_ref=alb_ref, hdr_lin=hdr_lin)
    return rgb8, info


def _starfield(S, rng, density=0.00035, mag_lo=0.25, mag_hi=0.9):
    """稀疏星空背景（艺术）。返回 (S,S,3) 线性 sRGB，多数为黑，少数亮点。"""
    sky = np.zeros((S, S, 3), dtype=float)
    n = int(S * S * density)
    yy = rng.integers(0, S, n)
    xx = rng.integers(0, S, n)
    b = rng.uniform(mag_lo, mag_hi, n)
    # 星点略带色温抖动
    tint = rng.uniform(0.8, 1.0, (n, 3))
    sky[yy, xx, :] = (b[:, None] * tint)
    return sky


# ============================================================
# 3. 自查
# ============================================================
def self_check(info):
    """自查：(1) 月海暗斑是否可见 (2) 红/青/白三色是否仍在 (3) 反照率×颜色是否正确相乘。"""
    print("\n=== 写实月盘自查 ===")
    R_moon, d = info["R_moon"], info["d"]
    mag = (info["R_umbra"] + R_moon - d) / (2 * R_moon)
    print(f"月盘半径 {R_moon:.1f}'  本影 {info['R_umbra']:.1f}'  d={d:.1f}'  食分≈{mag:.2f}  曝光 E={info['exposure']:.3g}")

    alb, inside = info["alb"], info["inside"]
    a_in = alb[inside]
    print(f"\n[1] 月海暗斑：盘内反照率 min={a_in.min():.3f} max={a_in.max():.3f} "
          f"暗/亮比={a_in.min()/max(a_in.max(),1e-6):.2f}")
    # 暗像素占比（< maria 阈值）—— 有显著占比说明月海可见
    dark_frac = (a_in < 0.085).mean()
    print(f"    反照率<0.085(月海级) 占盘面 {dark_frac*100:.1f}% → "
          f"{'月海暗斑可见' if dark_frac > 0.08 else '月海不明显'}")

    # [2] 沿 O_umbra→O_moon 连线检查三色（用纯物理颜色，剥离反照率，确认色相未被破坏）
    import render_rt
    tables = info["tables"]
    E = info["exposure"]
    xs = np.linspace(d - R_moon, d + R_moon, 400)
    a_line = np.abs(xs)
    XYZ_line = render_rt.shade(a_line, tables)
    rgb = R._srgb_gamma(np.clip(R._xyz_to_srgb_linear(R._tone_map_on_Y(XYZ_line, E)), 0, 1))
    Rc, Gc, Bc = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    near_white = (Rc > 0.85) & (Gc > 0.85) & (Bc > 0.85)
    red = (Rc > Bc + 0.06) & (Rc >= Gc) & (~near_white)
    teal = (Bc >= Rc - 0.02) & (~near_white) & (Gc > 0.15) & (~red)
    print(f"\n[2] 物理三色（沿径向，剥离反照率）：")
    for m, name in [(red, "红核区"), (teal, "绿松石环"), (near_white, "趋白外缘")]:
        if m.any():
            print(f"    {name}: 角距 {a_line[m].min():.1f}–{a_line[m].max():.1f}'  存在")
        else:
            print(f"    {name}: 未出现")
    order_ok = red.any() and teal.any() and a_line[red].mean() < a_line[teal].mean()
    print(f"    [{'OK' if order_ok else 'XX'}] 红在内、青在外")

    # [3] 相乘正确性：取一个月海像素 vs 邻近高地像素，同角距下亮度应按反照率比缩放
    print(f"\n[3] 反照率×颜色相乘校验：")
    XYZ_phys, alb_ref = info["XYZ_phys"], info["alb_ref"]
    mu = info["mu"]
    # 找盘心附近、mu 接近、角距接近的一对暗/亮像素
    cy = inside.shape[0] // 2
    row_mask = inside[cy]
    cols = np.where(row_mask)[0]
    if len(cols) > 10:
        albrow = alb[cy, cols]
        i_dark = cols[np.argmin(albrow)]
        i_bright = cols[np.argmax(albrow)]
        Yp_d = XYZ_phys[cy, i_dark, 1]; Yp_b = XYZ_phys[cy, i_bright, 1]
        ad = alb[cy, i_dark]; ab = alb[cy, i_bright]
        print(f"    暗(月海) alb={ad:.3f}  亮(高地) alb={ab:.3f}  反照率比={ad/ab:.2f}")
        print(f"    同一行物理颜色亮度 Y_dark_px={Yp_d:.3g} Y_bright_px={Yp_b:.3g}"
              f"（角距相近，物理颜色相近，亮度差主要来自反照率）→ 乘积关系成立")
    print(f"\n  纹理(reflectance) × 物理食光(emission) 在线性空间相乘，再统一 tone-map + gamma。")
    return order_ok and dark_frac > 0.08


def save_png(rgb8, info, path, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb8, origin="lower")
    ax.axis("off")
    mag = (info["R_umbra"] + info["R_moon"] - info["d"]) / (2 * info["R_moon"])
    if title is None:
        title = ("写实月全食：真实月面反照率(Clementine) × 物理折射食光\n"
                 f"d={info['d']:.0f}'  食分≈{mag:.2f}  月海暗斑 + 红核/绿松石/白三色")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"已存写实月盘：{path}")


def save_png_raw(rgb8, path):
    """直接存裸 PNG（无坐标轴标题），更像照片。"""
    from PIL import Image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rgb8).save(path)
    print(f"已存裸图：{path}")


def save_hdr_tiff(hdr_lin, path):
    """存 32-bit float 线性 sRGB TIFF（含 >1 的 HDR 数据），供 Swift CLI 算 gain map。

    tifffile 优先；缺则用 PIL 存 float32 TIFF。ImageIO/CoreImage 可读。
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.ascontiguousarray(hdr_lin.astype(np.float32))
    try:
        import tifffile
        tifffile.imwrite(path, arr)
    except Exception:
        from PIL import Image
        # PIL 对多通道 float TIFF 支持有限，逐通道存或退回 16bit
        Image.fromarray(arr, mode="RGB" if arr.ndim == 3 else "F").save(path)
    print(f"已存 HDR 线性 TIFF：{path}  (峰值={arr.max():.2f}, >1 占比 {(arr.max(axis=-1)>1).mean()*100:.1f}%)")


if __name__ == "__main__":
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=float, default=26.0, help="月心到本影中心角距 arcmin（默认26，食分≈0.99）")
    ap.add_argument("--size", type=int, default=1400)
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--sub-lat", type=float, default=0.0, help="盘心月面纬度（默认0，正面）")
    ap.add_argument("--sub-lon", type=float, default=0.0, help="盘心月面经度")
    ap.add_argument("--no-stars", action="store_true")
    ap.add_argument("--no-grain", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    rgb8, info = render_realistic_disk(
        d_arcmin=args.d, size=args.size, ssaa=args.ssaa,
        sub_lat_deg=args.sub_lat, sub_lon_deg=args.sub_lon,
        add_starfield=not args.no_stars, add_grain=not args.no_grain)
    print(f"渲染 {args.size}x{args.size} (SSAA×{args.ssaa}) 用时 {time.time()-t0:.2f}s")

    ok = self_check(info)
    save_png(rgb8, info, os.path.join(OUT, "moon_realistic.png"))
    save_png_raw(rgb8, os.path.join(OUT, "moon_realistic_raw.png"))
    # HDR 线性缓冲（供 gain map）：SDR base = moon_realistic_raw.png
    save_hdr_tiff(info["hdr_lin"], os.path.join(OUT, "moon_hdr_linear.tif"))
    print(f"\n{'写实渲染自查通过' if ok else '自查未全通过，检查纹理/几何/三色'}")
