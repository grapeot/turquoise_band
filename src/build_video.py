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

N_FRAMES = 300            # 从 D=10' 开始(跳过 D=0-10' 那段月盘全深本影、变化小/反直觉)
D_MIN, D_MAX = 10.0, 60.0
PANEL = 540          # 每半边像素
SSAA = 2

# 预建物理表（复用）
print("建物理表...")
MOON_T = render_rt.build_branch_tables(n_h=8000, use_focus=False)  # 关聚焦去亮斑
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

def _panel_tonemap(rgb_lin, gamma, target, knee=1.0):
    """每panel独立 SDR tone map(无floor, 时间全局定死, 不per-frame)。

    分段曲线(替代旧的 gamma+对数肩部——那套把对比压平、高光永不到白):
    - 暗部(Y<=knee): gamma 提亮保暗部层次/对比, knee 处亮度=target
    - 亮部(Y>knee):  线性快速冲到 1(白)——高光能真正 clip, 太阳露出时大气 bloom 到白
    这样既保住 spatial 对比(月球右亮左暗), 又让高光饱和(钻石环/大气 bloom)。
    色相保持: 按亮度映射比缩放 RGB。
    """
    Y = np.maximum(0.2126*rgb_lin[...,0]+0.7152*rgb_lin[...,1]+0.0722*rgb_lin[...,2], 1e-12)
    yk = target * np.power(np.minimum(Y, knee), gamma) / np.power(knee, gamma)  # 暗部gamma
    over = np.maximum(Y - knee, 0.0)
    Yd = np.clip(yk + over * (1.0 - target) / knee, 0.0, 1.0)  # 过knee线性冲白, 再涨1个knee到1
    scale = (Yd / Y)[..., None]
    return R._srgb_gamma(np.clip(rgb_lin * scale, 0, 1))


# ============================================================================
# 三个 panel 的 tone map 完全解耦：每个 panel 一套独立参数(SDR gain/gamma/target,
# HDR exp/black/white), 互不引用。调任一 panel 不影响另外两个。
# SDR _panel_tonemap: gamma 越小暗部越亮; target=正常亮度(Y=knee)映到的显示值,
#   留(1-target)余量给高光冲白; gain=tone map前线性曝光增益。
# HDR _panel_to_nits:  nits = black + exp·Y·white。black=暗部floor, white=正常亮度对应nits。
# ----------------------------------------------------------------------------
# [月球 panel] 接近线性保大对比(右出本影亮/左深本影血月暗), 暗部仍有层次
MOON_GAIN,  MOON_GAMMA,  MOON_TGT  = 1.0,  0.80, 0.78
MOON_HDR_EXP, MOON_HDR_BLACK, MOON_HDR_WHITE = 1.0, 0.8, 200.0   # 血月floor低(橙色不过亮), 正常月光200
# [地球全景 panel] 夜面暗于环(夜面亮度见 render_earth 夜面系数), 钻石环/太阳冲白
FULL_GAIN,  FULL_GAMMA,  FULL_TGT  = 1.0,  0.55, 0.72
FULL_HDR_EXP, FULL_HDR_BLACK, FULL_HDR_WHITE = 1.0, 0.0, 200.0   # 钻石环已超亮(~2000nits)
# [大气特写 panel] 环颜色梯度+高光bloom到白; HDR后期更亮(闪光感)
CLOSE_GAIN, CLOSE_GAMMA, CLOSE_TGT = 1.2,  0.62, 0.70
CLOSE_HDR_EXP, CLOSE_HDR_BLACK, CLOSE_HDR_WHITE = 1.0, 0.0, 420.0
# ============================================================================

# (旧的全局 gamma 月面曝光, render_textured 仍用)
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
    # 月球接近朗伯体, limb darkening 很弱(不像太阳那么强)。用 z^0.15 让满月较均匀,
    # 避免"出本影后右半边因边缘变暗显得不均匀"(用户反馈)。
    limb = np.power(np.clip(z,0,1), 0.15)
    # HDR 线性场景值(物理真实, 含全动态范围): 食光 × 反照率 × limb, 未压缩未tone-map。
    # 这是后期 HDR 处理的最大信息量来源(血月最深~6e-5 到 正常月光~1, 1.5万倍真实范围)。
    XYZ_hdr = XYZ * (albn*limb)[...,None]
    rgb_hdr = np.clip(R._xyz_to_srgb_linear(XYZ_hdr), 0, None) * inside[...,None]
    if hdr:
        return _box(rgb_hdr, SSAA)
    # 显示版: 月球独立 tone map(参数完全独立, 不与地球两panel共享)。
    rgb = _panel_tonemap(rgb_hdr * MOON_GAIN, MOON_GAMMA, MOON_TGT)
    rgb = rgb * inside[..., None]
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
    lin = RE.render_earth_frame(D, size=PANEL, ssaa=SSAA, earth_tex=EARTH_TEX,
                                ring_tables=RING_T, fov=None,   # 看整个地球
                                center=None, sun_dir_deg=180.0,
                                return_linear=True, draw_sun=True)   # 全景画太阳(钻石环)
    if hdr:
        return lin
    # 地球全景独立 tone map(不被月球曝光带着走→不过亮); 再加亮一半
    out = _panel_tonemap(lin * FULL_GAIN, FULL_GAMMA, FULL_TGT)
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
    # 特写看底部弧(水平地平线,干净)。太阳也朝底部(-90)让这段随D演变(太阳贴近→变亮)。
    # 注: 全景太阳在左缘(物理对,配月亮右移); 特写是"放大看环颜色"非方位图, 用底部好看。
    ring_mid = RE.ANG_EARTH * (1.0 + RE.RING_FRAC * 0.5)
    lin = RE.render_earth_frame(D, size=PANEL, ssaa=SSAA, earth_tex=EARTH_TEX,
                                ring_tables=RING_T, fov=3.0,
                                center=(0.0, -ring_mid), sun_dir_deg=-90.0,
                                return_linear=True, draw_sun=False)
    out = lin if hdr else _panel_tonemap(lin * CLOSE_GAIN, CLOSE_GAMMA, CLOSE_TGT)  # 特写独立参数
    return out[::-1]   # 上下翻转: 天空在上、地球在下(自然地平线)


def _pq_encode(linear_nits):
    """SMPTE2084 PQ 传递函数: 线性亮度(nits) → [0,1] PQ 码值。"""
    L = np.clip(linear_nits, 0, 10000) / 10000.0
    m1, m2 = 0.1593017578125, 78.84375
    c1, c2, c3 = 0.8359375, 18.8515625, 18.6875
    Lm = np.power(L, m1)
    return np.power((c1 + c2 * Lm) / (1 + c3 * Lm), m2)


def _panel_to_nits(rgb_lin, exposure, black_nits, white_nits):
    """每panel独立: 线性场景值 → nits。暗部抬到black_nits保底可见, 亮部线性不封顶。
      nits = black_nits + (exposure·Y) × white_nits   (Y已是线性, 不压gamma→保亮度增量)
    暗部(Y小)抬到~black_nits; 正常(exposure·Y~1)→~white_nits; 亮部(Y大)线性超亮不封顶。
    """
    Y = np.maximum(0.2126*rgb_lin[...,0]+0.7152*rgb_lin[...,1]+0.0722*rgb_lin[...,2], 1e-12)
    nits_Y = black_nits + exposure * Y * white_nits
    scale = (nits_Y / Y)[..., None]
    return rgb_lin * scale


def _render_hdr_frame(args):
    """渲一帧 HDR: 三panel各自独立线性→nits映射(像SDR那样独立), 拼接后PQ编码。"""
    import tifffile
    i, D = args
    # 三panel各自线性HDR
    moon = render_moon_panel(D, hdr=True)
    full = render_earth_full_panel(D, hdr=True)
    close = render_earth_panel(D, hdr=True)
    gap = np.zeros((PANEL, 6, 3), np.float32)
    # (1) 线性 TIFF(后期用): 拼接线性帧, 固定scale。
    lin = np.concatenate([moon, gap, full, gap, close], axis=1).astype(np.float32)
    _add_hdr_markers(lin)
    f16_lin = np.clip(lin * 16000.0, 0, 65535).astype(np.uint16)
    tifffile.imwrite(os.path.join(HDRDIR, f"f{i:04d}.tif"), f16_lin)
    # (2) HDR视频: 每panel独立映射到nits(地球不被月球带亮、保各自亮度增量), 拼接后PQ。
    moon_n = _panel_to_nits(moon, MOON_HDR_EXP, MOON_HDR_BLACK, MOON_HDR_WHITE)    # 月球独立
    full_n = _panel_to_nits(full, FULL_HDR_EXP, FULL_HDR_BLACK, FULL_HDR_WHITE)    # 全景独立
    close_n = _panel_to_nits(close, CLOSE_HDR_EXP, CLOSE_HDR_BLACK, CLOSE_HDR_WHITE)  # 特写独立
    nits = np.concatenate([moon_n, gap, full_n, gap, close_n], axis=1)
    _add_hdr_markers(nits, val=MOON_HDR_WHITE)         # marker用白点nits亮度
    pq = _pq_encode(nits)
    pq16 = (np.clip(pq, 0, 1) * 65535 + 0.5).astype(np.uint16)
    # 角分文字: 8bit渲染文字蒙版, 叠到PQ帧(PIL不支持16bit RGB绘字, 故用蒙版)。
    txt = _angle_text_mask(D, pq16.shape[:2])
    pq16[txt] = 52000                                  # 文字处置亮白(PQ码值)
    tifffile.imwrite(os.path.join(HDRDIR, f"pq{i:04d}.tif"), pq16)
    return i


_TXT_FONT = None
def _angle_text_mask(D, shape):
    """返回角分文字的布尔蒙版(在8bit灰度图上渲染文字再阈值化)。"""
    global _TXT_FONT
    from PIL import ImageDraw, ImageFont
    if _TXT_FONT is None:
        try:
            _TXT_FONT = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", 22)
        except Exception:
            _TXT_FONT = ImageFont.load_default()
    g = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(g).text((14, 10), f"月心距本影中心 D = {D:.1f}'", fill=255, font=_TXT_FONT)
    return np.asarray(g) > 128


def _add_hdr_markers(frame, val=2.0):
    """在三栏HDR帧上加月面观测点三角+全景特写框。val=marker亮度(线性帧用2, nits帧用白点nits)。"""
    H, W3 = frame.shape[:2]
    pw = (W3 - 12) // 3                                # 每栏宽(含gap)
    cyp = H // 2; cxp = pw // 2
    for dy in range(8):                                # 月面观测点三角(青)
        half = 8 - dy
        frame[max(0,cyp-8+dy), max(0,cxp-half):cxp+half] = [0.0, val, val]
    px0 = pw + 6                                       # 全景左缘特写框(黄)
    cxp2 = px0 + int((-RE.ANG_EARTH + RE.ANG_EARTH*1.15) / (2*RE.ANG_EARTH*1.15) * pw)
    bs = 22
    for t in range(-bs, bs+1):
        for e in [-bs, bs]:
            frame[np.clip(cyp+e,0,H-1), np.clip(cxp2+t,0,W3-1)] = [val, val, 0.0]
            frame[np.clip(cyp+t,0,H-1), np.clip(cxp2+e,0,W3-1)] = [val, val, 0.0]


def _draw_angle(rgb8, D):
    """左上角标月心距本影中心角分数值。rgb8: uint8 (H,W,3)。"""
    from PIL import ImageDraw, ImageFont
    im = Image.fromarray(rgb8)
    d = ImageDraw.Draw(im)
    font = None
    for fp in ["/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
               "/System/Library/Fonts/STHeiti Medium.ttc"]:
        try:
            font = ImageFont.truetype(fp, 22); break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    d.text((14, 10), f"月心距本影中心 D = {D:.1f}'", fill=(255, 255, 255), font=font)
    return np.asarray(im)


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
    f8 = _draw_angle(f8, D)
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
        subprocess.run(["ffmpeg", "-y", "-framerate", "60", "-i", os.path.join(FRAMEDIR, "f%04d.png"),
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
            "ffmpeg", "-y", "-framerate", "60", "-i", os.path.join(HDRDIR, "pq%04d.tif"),
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
