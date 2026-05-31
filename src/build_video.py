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


def render_moon_panel(D):
    """左panel：月盘在距本影中心 D 处，食光颜色×月面纹理。忠实亮度。"""
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
    lo, hi = np.percentile(albY[inside], 5), np.percentile(albY[inside], 95)
    albn = np.clip((albY-lo)/max(hi-lo,1e-6)*0.5+0.5, 0.2, 1.2)
    limb = np.power(np.clip(z,0,1), 0.5)
    XYZ_scene = XYZ * (albn*limb)[...,None]
    # 忠实亮度曝光(按月盘最亮)
    Ys = XYZ_scene[...,1]
    Yb = np.percentile(Ys[inside], 99.5) if inside.any() else 1
    E = R._srgb_inv_gamma(0.92)/max(Yb,1e-12)
    rgb = R._srgb_gamma(np.clip(R._xyz_to_srgb_linear(R._tone_map_on_Y(XYZ_scene, E)),0,1))
    rgb = rgb*inside[...,None]
    return _box(rgb, SSAA)


def render_earth_panel(D):
    """右panel：站月面中心看地球，长焦看亮侧(底部)那段大气环。"""
    rgb8 = RE.render_earth_frame(D, size=PANEL, ssaa=SSAA, earth_tex=EARTH_TEX,
                                 ring_tables=RING_T, fov=20.0,
                                 center=(0.0, -RE.ANG_EARTH), sun_dir_deg=-90.0)
    return rgb8.astype(float)/255.0


def _box(img, f):
    if f == 1: return img
    s = img.shape[0]; n = s//f
    return img[:n*f,:n*f].reshape(n,f,n,f,img.shape[2]).mean(axis=(1,3))


def main():
    Ds = np.linspace(D_MIN, D_MAX, N_FRAMES)
    import time
    t0 = time.time()
    for i, D in enumerate(Ds):
        moon = render_moon_panel(D)
        earth = render_earth_panel(D)
        # 拼接：左月面 右地球，中间细分隔
        gap = np.zeros((PANEL, 6, 3))
        frame = np.concatenate([moon, gap, earth], axis=1)
        f8 = (np.clip(frame,0,1)*255+0.5).astype(np.uint8)
        Image.fromarray(f8).save(os.path.join(FRAMEDIR, f"f{i:04d}.png"))
        if i % 30 == 0:
            el = time.time()-t0
            print(f"  帧 {i}/{N_FRAMES} D={D:.1f}'  ({el:.0f}s, ~{el/(i+1)*N_FRAMES:.0f}s total)")
    print(f"渲染完成 {time.time()-t0:.0f}s. 合成视频...")
    # ffmpeg 合成
    out_mp4 = os.path.join(OUT, "moon_eclipse_dual.mp4")
    subprocess.run(["ffmpeg","-y","-framerate","30","-i",os.path.join(FRAMEDIR,"f%04d.png"),
                    "-c:v","libx264","-pix_fmt","yuv420p","-crf","18", out_mp4], check=True)
    print(f"已合成 {out_mp4}")


if __name__ == "__main__":
    main()
