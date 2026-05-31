"""合成月食视频：左=月面(月盘随D移出本影)，右=站月面中心看地球(大气环随D演变)。

360 帧，月心距本影中心 D 从 0→60 arcmin。逐帧渲染拼接，ffmpeg 合成 mp4。
物理表只建一次复用（月面 render_rt LUT + 地球环 LUT）。
"""
import os
import sys
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import render_rt
import render as R
import render_earth as RE
import geometry as g
from PIL import Image

OUT = R.OUT
FRAMEDIR = os.path.join(OUT, "video_frames")
HDRDIR = os.path.join(OUT, "video_frames_hdr")
os.makedirs(FRAMEDIR, exist_ok=True)

N_FRAMES = 360
D_MIN, D_MAX = 0.0, 60.0
PANEL = 540          # 每半边像素
SSAA = 2

# 预建物理表（复用）
print("建物理表...")
MOON_T = render_rt.build_branch_tables(n_h=8000)
RING_T = RE._ring_color_table()
# 月面纹理 + 地球夜面
Image.MAX_IMAGE_PIXELS = None
_td = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
_moon_tex_p = os.path.join(_td, "moon_texture", "nasa_moon_color_lroc_4k_16bit.tif")
MOON_TEX = np.asarray(Image.open(_moon_tex_p).convert("RGB"), float)
MOON_TEX = MOON_TEX / max(MOON_TEX.max(), 1)
_earth_p = os.path.join(_td, "earth_texture", "earth_blackmarble_2016_3600x1800.jpg")
EARTH_TEX = np.asarray(Image.open(_earth_p).convert("RGB"), float) / 255.0 if os.path.exists(_earth_p) else None

R_MOON = R.R_MOON_ARCMIN
R_UMBRA = R.R_UMBRA_ARCMIN

# 全局固定月面曝光：正常月光(shade 返回 Y=1) × 反照率(~0.6中性) × limb 后映到 sRGB~0.92。
# 不随帧变——保留"月盘随移出本影逐渐变亮"的真实演变。
# 全局动态范围压缩 gamma(固定不随帧): 血月最深~6e-5 与正常月光~1 差1.5万倍,
# Y^DYN_GAMMA 把这压进可见范围。越小压得越狠(血月越亮)。0.35 让血月落暗红可见。
DYN_GAMMA = 0.35
_Y_NORMAL = 1.0 * 0.6              # 正常月光×中性反照率(albn≈0.5-0.6)
MOON_E = R._srgb_inv_gamma(0.75) / (_Y_NORMAL ** DYN_GAMMA)
# 全月面反照率全局百分位(固定, 不随帧变)
_mY = 0.2126*MOON_TEX[...,0]+0.7152*MOON_TEX[...,1]+0.0722*MOON_TEX[...,2]
MOON_ALB_LO, MOON_ALB_HI = np.percentile(_mY, 5), np.percentile(_mY, 95)


def render_moon_panel(D, hdr=False, mark=True):
    """左panel：月盘在距本影中心 D 处，食光颜色×月面纹理。
    hdr/mark 见下。mark=True 月心标观测点三角(我们站这看地球)。"""
    S = PANEL * SSAA
    # 画幅固定看月盘(以月心为中心)
    half = R_MOON * 1.15
    cx = D
    xs = np.linspace(cx - half, cx + half, S)
    ys = np.linspace(-half, half, S)
    X, Y = np.meshgrid(xs, ys)
    a = np.hypot(X, Y)                         # 到本影中心角距
    rmoon = np.hypot(X - cx, Y)
    inside = rmoon <= R_MOON
    XYZ = render_rt.shade(a, MOON_T)
    # 月面纹理(正交投影) × 食光
    U = (X - cx) / R_MOON; V = Y / R_MOON
    z = np.sqrt(np.clip(1 - U*U - V*V, 0, 1))
    lat = np.arcsin(np.clip(V, -1, 1)); lon = np.arctan2(U, z)
    Ht, Wt = MOON_TEX.shape[:2]
    ci = np.clip(((np.degrees(lon)+180)/360*(Wt-1)).astype(int), 0, Wt-1)
    ri = np.clip(((90-np.degrees(lat))/180*(Ht-1)).astype(int), 0, Ht-1)
    alb = MOON_TEX[ri, ci]
    albY = (0.2126*alb[...,0]+0.7152*alb[...,1]+0.0722*alb[...,2])
    # 反照率归一用**全月面**全局百分位(MOON_ALB_LO/HI)，不随帧/月盘位置变——
    # 否则每帧采到不同经度月面、反照率分布不同会让月盘平均亮度跳变。
    albn = np.clip((albY-MOON_ALB_LO)/max(MOON_ALB_HI-MOON_ALB_LO,1e-6)*0.5+0.5, 0.2, 1.2)
    limb = np.power(np.clip(z,0,1), 0.5)
    # HDR 线性场景值(物理真实, 含全动态范围): 食光 × 反照率 × limb, 未压缩未tone-map。
    # 这是后期 HDR 处理的最大信息量来源(血月最深~6e-5 到 正常月光~1, 1.5万倍真实范围)。
    XYZ_hdr = XYZ * (albn*limb)[...,None]
    rgb_hdr = np.clip(R._xyz_to_srgb_linear(XYZ_hdr), 0, None) * inside[...,None]
    if hdr:
        return _box(rgb_hdr, SSAA)
    # 显示版(8-bit预览): 动态范围压缩(全局固定 gamma) + 固定曝光 + tone-map。
    Yp = np.maximum(XYZ[...,1], 1e-30)
    chroma_xyz = XYZ / Yp[...,None]
    Y_comp = np.power(Yp, DYN_GAMMA)
    XYZ_comp = chroma_xyz * Y_comp[...,None]
    XYZ_scene = XYZ_comp * (albn*limb)[...,None]
    rgb = R._srgb_gamma(np.clip(R._xyz_to_srgb_linear(R._tone_map_on_Y(XYZ_scene, MOON_E)),0,1))
    rgb = rgb*inside[...,None]
    out = _box(rgb, SSAA)
    if mark:
        # 观测点 marker: 我们站在月盘中心看地球。标个小三角(尖朝上)在月心。
        H, W = out.shape[:2]
        # 月心在画幅: cx=D是月盘中心(world x), 画幅以cx为中心→月心在图中央
        cyp = H // 2; cxp = W // 2
        for dy in range(10):
            half = (10 - dy)
            out[max(0, cyp - 10 + dy), max(0, cxp - half):min(W, cxp + half)] = [0.2, 1.0, 1.0]
    return out


def render_earth_full_panel(D, hdr=False, mark=True):
    """中panel：地球全景(整个地球盘+细大气环+太阳)，底部标三角=右栏特写看的位置。"""
    rgb = RE.render_earth_frame(D, size=PANEL, ssaa=SSAA, earth_tex=EARTH_TEX,
                                ring_tables=RING_T, fov=None,   # 看整个地球
                                center=None, sun_dir_deg=180.0,
                                return_linear=hdr, draw_sun=True)   # 全景画太阳(钻石环)
    out = rgb if hdr else rgb.astype(float)/255.0
    if mark and not hdr:
        # 在地球**左缘**(太阳露出侧/特写取景处)画一个小方框，指出右栏特写看的是这段环。
        H, W = out.shape[:2]
        half_world = RE.ANG_EARTH * 1.15
        # 左缘世界坐标(-ANG_EARTH, 0) → 像素。x: world→col, y朝上=图上→row。
        cxp = int((-RE.ANG_EARTH + half_world) / (2*half_world) * (W-1))
        cyp = H // 2
        bs = 26                                      # 框半边(像素)
        c = [1.0, 1.0, 0.2]
        for t in range(-bs, bs+1):
            for e in [-bs, bs]:
                out[np.clip(cyp+e,0,H-1), np.clip(cxp+t,0,W-1)] = c   # 上下边
                out[np.clip(cyp+t,0,H-1), np.clip(cxp+e,0,W-1)] = c   # 左右边
    return out


def render_earth_panel(D, hdr=False):
    """右panel：站月面中心看地球，长焦看亮侧(底部)那段大气环。
    hdr=True 返回线性HDR(地球环高动态)；否则8bit显示。"""
    # 真实薄环(1.3%)→小fov长焦放大看清; 弧因只看地球缘极小一段而趋平(像ISS看地平线)。
    # 特写不标方向(只看颜色梯度), 故构图旋成水平地平线最好看: 看底部弧、太阳从底部侧(亮侧)、
    # 不画太阳。与全景(左缘+太阳)方向不一致无妨——特写是"放大看这段环的颜色", 非方位图。
    ring_mid = RE.ANG_EARTH * (1.0 + RE.RING_FRAC * 0.5)
    rgb = RE.render_earth_frame(D, size=PANEL, ssaa=SSAA, earth_tex=EARTH_TEX,
                                ring_tables=RING_T, fov=3.0,
                                center=(0.0, -ring_mid), sun_dir_deg=-90.0,
                                return_linear=hdr, draw_sun=False)
    out = rgb if hdr else rgb.astype(float)/255.0
    # 上下翻转成自然地平线观感: 天空在上、地球在下(像站地面看日出地平线)。
    return out[::-1]


# 正常月光的参考亮度(nits)。地球环/太阳是高光(>这个值, 在HDR里超亮)。
HDR_WHITE_NITS = 200.0


def _pq_encode(linear_nits):
    """SMPTE2084 PQ 传递函数: 线性亮度(nits) → [0,1] PQ 码值。"""
    L = np.clip(linear_nits, 0, 10000) / 10000.0
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Lm = np.power(L, m1)
    return np.power((c1 + c2 * Lm) / (1 + c3 * Lm), m2)


def _render_hdr_frame(args):
    """渲一帧: (1)16bit线性TIFF(后期用) (2)16bit PQ-PNG(给HDR10视频)。三栏线性HDR。"""
    import tifffile
    i, D = args
    frame = _assemble(D, hdr=True).astype(np.float32)   # 线性场景值(正常月光≈1, 地球环/太阳>1)
    # (1) 线性 TIFF: 固定 scale 保留相对HDR(正常月光1.0→16000, 留headroom)。
    f16_lin = np.clip(frame * 16000.0, 0, 65535).astype(np.uint16)
    tifffile.imwrite(os.path.join(HDRDIR, f"f{i:04d}.tif"), f16_lin)
    # (2) PQ 编码给视频: 线性场景值→nits(正常月光=HDR_WHITE_NITS)→PQ→16bit RGB TIFF。
    # (PIL 不支持16bit RGB PNG, 用 tifffile 存16bit RGB; ffmpeg 能读 rgb48 tiff)
    nits = frame * HDR_WHITE_NITS
    pq = _pq_encode(nits)
    pq16 = (np.clip(pq, 0, 1) * 65535 + 0.5).astype(np.uint16)
    tifffile.imwrite(os.path.join(HDRDIR, f"pq{i:04d}.tif"), pq16)
    return i


def _box(img, f):
    if f == 1: return img
    s = img.shape[0]; n = s//f
    return img[:n*f,:n*f].reshape(n,f,n,f,img.shape[2]).mean(axis=(1,3))


def _assemble(D, hdr=False):
    """三栏: 左月球 | 中地球全景(带三角标观测点) | 右大气环长焦特写。"""
    moon = render_moon_panel(D, hdr=hdr)
    full = render_earth_full_panel(D, hdr=hdr)
    close = render_earth_panel(D, hdr=hdr)
    gap = np.zeros((PANEL, 6, 3))
    return np.concatenate([moon, gap, full, gap, close], axis=1)


def _render_one(args):
    """渲一帧 PNG(SDR显示版)。物理表/纹理是模块级全局，fork 子进程继承。"""
    i, D = args
    frame = _assemble(D, hdr=False)
    f8 = (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(f8).save(os.path.join(FRAMEDIR, f"f{i:04d}.png"))
    return i


def _run_parallel(fn, tasks, workers, label):
    import multiprocessing as mp, time
    ctx = mp.get_context("fork")
    n = workers or max(1, min(mp.cpu_count() - 2, 30))
    print(f"{label}: 并行 {len(tasks)} 帧, {n} 进程...")
    t0 = time.time()
    with ctx.Pool(n) as pool:
        for k, _ in enumerate(pool.imap_unordered(fn, tasks, chunksize=4)):
            if k % 60 == 0:
                print(f"  {k}/{len(tasks)} ({time.time()-t0:.0f}s)")
    print(f"  完成 {time.time()-t0:.0f}s")


def main(workers=None, do_png=True, do_tiff=True):
    Ds = np.linspace(D_MIN, D_MAX, N_FRAMES)
    tasks = list(enumerate(Ds))

    if do_png:
        _run_parallel(_render_one, tasks, workers, "PNG(SDR)")
        out = os.path.join(OUT, "moon_eclipse_sdr_h265.mp4")
        # SDR H.265
        subprocess.run(["ffmpeg", "-y", "-framerate", "30", "-i", os.path.join(FRAMEDIR, "f%04d.png"),
                        "-c:v", "libx265", "-pix_fmt", "yuv420p", "-crf", "20",
                        "-tag:v", "hvc1", out], check=True)
        print(f"SDR H.265: {out}")

    if do_tiff:
        os.makedirs(HDRDIR, exist_ok=True)
        _run_parallel(_render_hdr_frame, tasks, workers, "TIFF(HDR)")
        out = os.path.join(OUT, "moon_eclipse_hdr_h265.mp4")
        # HDR H.265: PQ 编码已在 Python 完成(pq*.tif 是 PQ 码值), ffmpeg 只转10bit+打HDR10标记。
        # 不依赖 zscale。BT.2020 primaries + SMPTE2084(PQ) transfer。
        subprocess.run([
            "ffmpeg", "-y", "-framerate", "30", "-i", os.path.join(HDRDIR, "pq%04d.tif"),
            "-vf", "format=gbrp16le,format=yuv420p10le",
            "-c:v", "libx265", "-crf", "18", "-tag:v", "hvc1",
            "-color_primaries", "bt2020", "-color_trc", "smpte2084", "-colorspace", "bt2020nc",
            "-x265-params",
            "hdr-opt=1:repeat-headers=1:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:"
            "master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1):max-cll=1000,200",
            out], check=True)
        print(f"HDR H.265: {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--png-only", action="store_true")
    ap.add_argument("--tiff-only", action="store_true")
    args = ap.parse_args()
    main(workers=args.workers,
         do_png=not args.tiff_only, do_tiff=not args.png_only)
