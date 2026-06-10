"""【渲染应用】真实星场: 用 Yale Bright Star Catalog 的真实亮星(真位置+真星等+真星色)生成背景。

注意: 当前没有任何管线引用本模块(render_textured 用的是自己的简化 _starfield),
保留作可复用素材(视频/ablation/艺术图若需要真实星场可接入)。

物理正确的两件事(用户要求):
1. **真实星等分布**: 直接用真实亮星表(不是椒盐随机), 越亮越稀疏自然成立。只用亮星
   (Vmag<6, 比月亮暗很多的暗星没价值, 用户判断)。
2. **真实相对亮度**: 星等→亮度 L=10^(-0.4(m−m_moon)), 锚满月 −12.7 等。这样星和月面
   的相对亮度真实——满月旁星几乎不可见, 月食暗部(古铜血月−13档)旁亮星可比/可见。
   **作绝对亮度 anchor**: 同一星场叠到各步, 星vs月面相对可见度反映月面绝对亮度变化。

天区: 月食必在黄道反日点(月亮黄纬≈0), 故取黄道带真实天区(默认金牛座, 有毕宿五Aldebaran
+昴星团, 视觉丰富)。不能用高赤纬亮星(物理上月食天区在黄道)。

模块缓存星表到 data/raw/。可复用: 视频/ablation/艺术图 共用。
"""
import os
import numpy as np

_CACHE = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "bsc_ecliptic.npz")
M_MOON = -12.7      # 满月视星等(锚)


def _fetch_catalog(ra_deg=67.0, dec_deg=18.0, radius_deg=22.0, vmag_max=6.0):
    """从 VizieR 拉 Yale BSC(V/50) 指定天区的亮星。缓存到 npz。"""
    if os.path.exists(_CACHE):
        d = np.load(_CACHE)
        return d["ra"], d["dec"], d["vmag"], d["bv"]
    from astroquery.vizier import Vizier
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    v = Vizier(columns=["RAJ2000", "DEJ2000", "Vmag", "B-V"],
               column_filters={"Vmag": f"<{vmag_max}"}, row_limit=5000)
    res = v.query_region(SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg"),
                         radius=radius_deg * u.deg, catalog="V/50")
    t = res[0]
    # RA/Dec 字符串→度
    from astropy.coordinates import Angle
    ra = Angle(t["RAJ2000"], unit=u.hourangle).deg
    dec = Angle(t["DEJ2000"], unit=u.deg).deg
    vmag = np.asarray(t["Vmag"], float)
    bv = np.nan_to_num(np.asarray(t["B-V"], float), nan=0.5)
    ok = np.isfinite(vmag)
    ra, dec, vmag, bv = ra[ok], dec[ok], vmag[ok], bv[ok]
    os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
    np.savez(_CACHE, ra=ra, dec=dec, vmag=vmag, bv=bv)
    return ra, dec, vmag, bv


def _bv_to_rgb(bv):
    """B-V 色指数 → 归一化 RGB 星色。蓝(B-V<0)→白(0.6)→橙红(>1.4)。subtle。"""
    r = np.clip(0.75 + bv * 0.28, 0.55, 1.18)
    b = np.clip(1.18 - bv * 0.42, 0.5, 1.2)
    g = np.clip(1.05 - np.abs(bv - 0.45) * 0.18, 0.6, 1.1)
    return np.stack([r, g, b], axis=-1)


def render_starfield(W, H, fov_deg, moon_L=1.0, ra_center=67.0, dec_center=18.0,
                     psf_px=0.8, vmag_max=6.0, boost=250.0):
    """渲染真实星场到 (H,W,3) 线性画布(相对满月亮度, 满月=moon_L)。

    fov_deg: 画幅水平视场(度)。ra/dec_center: 画幅中心天区(默认金牛座黄道带)。
    每颗真实亮星: 真位置(gnomonic投影到画幅)+真星等→亮度(锚满月)+真B-V→星色。
    boost: 整体亮度提升(让背景星空可见)。星**之间**相对亮度仍真实(星等分布不变),
      仅整体抬一个固定系数。boost=1 为纯物理(满月旁几乎不可见)。默认250让毕宿五≈月食暗部可比。
    """
    from scipy.ndimage import gaussian_filter
    ra, dec, vmag, bv = _fetch_catalog(ra_center, dec_center, vmag_max=vmag_max)
    # 星等→相对满月亮度(锚 M_MOON)
    L = moon_L * 10.0 ** (-0.4 * (vmag - M_MOON))
    cols = _bv_to_rgb(bv)

    # gnomonic(切平面)投影: 天区中心为切点, RA/Dec→画幅 x,y(度)
    ra0, dec0 = np.radians(ra_center), np.radians(dec_center)
    ar, dr = np.radians(ra), np.radians(dec)
    cosc = np.sin(dec0) * np.sin(dr) + np.cos(dec0) * np.cos(dr) * np.cos(ar - ra0)
    cosc = np.where(np.abs(cosc) < 1e-6, 1e-6, cosc)
    x = np.cos(dr) * np.sin(ar - ra0) / cosc
    y = (np.cos(dec0) * np.sin(dr) - np.sin(dec0) * np.cos(dr) * np.cos(ar - ra0)) / cosc
    xdeg = np.degrees(x); ydeg = np.degrees(y)

    # 画幅: fov_deg 宽, 居中。映射到像素
    scale = W / fov_deg
    px = (xdeg * scale + W / 2).astype(int)
    py = (H / 2 - ydeg * scale).astype(int)   # y 朝上
    inside = (px >= 0) & (px < W) & (py >= 0) & (py < H)

    canvas = np.zeros((H, W, 3), np.float32)
    np.add.at(canvas, (py[inside], px[inside]), (L[inside, None] * cols[inside] * boost))
    for c in range(3):
        canvas[..., c] = gaussian_filter(canvas[..., c], psf_px)  # 星点 PSF 光晕
    return canvas
