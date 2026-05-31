"""导出月食视频所需的全部物理数据（引擎无关），供 Unreal/Blender/任意渲染器使用。

视频：360 帧，月心距本影中心 D 从 0→60 arcmin，左=月面、右=地球(站月面中心看)。

导出物（outputs/video_data/）：
- moon_eclipse_color.csv   月面食光颜色 LUT：角距(arcmin) → 线性RGB（不随帧变）
- moon_eclipse_color.png   同上的 1D 渐变条（可直接当贴图采样）
- earth_ring_color.csv     地球大气环径向 LUT：擦边高度(km) → 线性RGB（不随帧变）
- earth_ring_color.png     同上渐变条
- frames.csv              每帧几何：frame, D_arcmin, 太阳偏移, 环亮度方位调制参数
- meta.json               角尺寸常数、资产清单、坐标约定、单位说明
全部线性 RGB（未 gamma），渲染器自行做 tone-map/gamma。
"""
import os
import sys
import json
import csv
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import render_rt
import render_earth as RE
import render as R
import geometry as g

OUTDIR = os.path.join(R.OUT, "video_data")
os.makedirs(OUTDIR, exist_ok=True)

N_FRAMES = 360
D_MIN, D_MAX = 0.0, 60.0


def export_moon_color_lut(n=512):
    """月面食光颜色：角距(距本影中心 arcmin) → 线性 RGB。复用 render_rt 物理。"""
    t = render_rt.build_branch_tables(n_h=8000)
    a = np.linspace(0, 60, n)                       # 角距 0..60'
    XYZ = render_rt.shade(a, t)
    rgb_lin = np.clip(R._xyz_to_srgb_linear(XYZ), 0, None)   # 线性 RGB（含真实亮度）
    # CSV
    with open(os.path.join(OUTDIR, "moon_eclipse_color.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arcmin_from_umbra_center", "R_lin", "G_lin", "B_lin", "Y_luminance"])
        for i in range(n):
            w.writerow([f"{a[i]:.4f}", f"{rgb_lin[i,0]:.6e}", f"{rgb_lin[i,1]:.6e}",
                        f"{rgb_lin[i,2]:.6e}", f"{XYZ[i,1]:.6e}"])
    # PNG 渐变条（gamma 后，便于肉眼检查；引擎用 CSV 的线性值）
    _save_strip(rgb_lin, os.path.join(OUTDIR, "moon_eclipse_color.png"))
    return a, rgb_lin


def export_earth_ring_lut(n=512):
    """地球大气环径向：擦边高度(km, 0→30) → 线性 RGB（环内边→外边 = 红→青绿→白）。"""
    h_grid, XYZ_grid = RE._ring_color_table()       # h_max=30km
    h = np.linspace(0, RE.H_ATM, n)
    XYZ = RE._interp_xyz(h, h_grid, XYZ_grid)
    rgb_lin = np.clip(R._xyz_to_srgb_linear(XYZ), 0, None)
    with open(os.path.join(OUTDIR, "earth_ring_color.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tangent_height_km", "R_lin", "G_lin", "B_lin"])
        for i in range(n):
            w.writerow([f"{h[i]:.4f}", f"{rgb_lin[i,0]:.6e}", f"{rgb_lin[i,1]:.6e}", f"{rgb_lin[i,2]:.6e}"])
    _save_strip(rgb_lin, os.path.join(OUTDIR, "earth_ring_color.png"))
    return h, rgb_lin


def export_frames():
    """每帧几何：D、太阳偏移、钻石环露出量。环方位亮度的物理公式见 meta.json。"""
    Ds = np.linspace(D_MIN, D_MAX, N_FRAMES)
    with open(os.path.join(OUTDIR, "frames.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "D_arcmin", "moon_center_x_arcmin",
                    "sun_offset_arcmin", "sun_peek_arcmin", "phase"])
        for i, D in enumerate(Ds):
            sun_peek = D + RE.ANG_SUN - RE.ANG_EARTH       # >0 = 太阳缘探出 = 钻石环
            if D < RE.ANG_EARTH - RE.ANG_SUN:
                phase = "total"        # 太阳全藏地球后，全圈日落环
            elif sun_peek < 0:
                phase = "deep"         # 太阳贴近边缘，一侧很亮
            elif D < RE.ANG_EARTH:
                phase = "diamond"      # 钻石环
            else:
                phase = "egress"       # 太阳露出
            w.writerow([i, f"{D:.4f}", f"{D:.4f}", f"{D:.4f}", f"{sun_peek:.4f}", phase])
    return Ds


def _save_strip(rgb_lin, path, h=48):
    from PIL import Image
    rgb = R._srgb_gamma(np.clip(rgb_lin / max(rgb_lin.max(), 1e-6), 0, 1))
    strip = np.tile((rgb * 255).astype(np.uint8)[None, :, :], (h, 1, 1))
    Image.fromarray(strip).save(path)


def export_meta():
    meta = {
        "description": "月食视频物理数据（引擎无关）。左=月面看地球本影投影，右=站月面中心看地球。",
        "n_frames": N_FRAMES, "D_range_arcmin": [D_MIN, D_MAX],
        "angular_sizes_arcmin": {
            "earth_radius": RE.ANG_EARTH, "sun_radius": RE.ANG_SUN,
            "moon_radius": float(R.R_MOON_ARCMIN), "umbra_radius": float(R.R_UMBRA_ARCMIN),
        },
        "ring": {
            "radial_maps_tangent_height_km": [0, RE.H_ATM],
            "true_thickness_frac": RE.RING_FRAC_TRUE, "display_thickness_frac": RE.RING_FRAC,
            "note": "环径向(内→外)=擦边高度0→30km=红→青绿→白(查 earth_ring_color)。真实厚度仅1.3%，展示放大~8×。",
        },
        "ring_azimuth_brightness": {
            "formula": "bright(phi) = exp(-((dist_to_sun - earth_radius)/soft)^2)*0.7 + 0.05; soft=2.5*sun_radius",
            "dist_to_sun": "limb点(earth_radius·cos phi, ·sin phi) 到 太阳中心(D·cos sun_dir, D·sin sun_dir) 的角距",
            "diamond_spike": "D+sun_radius>earth_radius 时朝太阳侧加强光; 见 render_earth.py",
            "note": "D=0太阳正中→全圈均匀(360°日落环); D增大→朝太阳侧亮、背侧暗; D>edge→钻石环。",
        },
        "moon_disk": {
            "color_lut": "moon_eclipse_color (角距→线性RGB)。月盘每像素按到本影中心角距查色。",
            "geometry": "月心在距本影中心 D 处(frames.csv); 月盘半径见 angular_sizes。",
        },
        "assets": {
            "moon_color": "data/raw/moon_texture/nasa_moon_color_lroc_4k_16bit.tif",
            "moon_displacement": "data/raw/moon_texture/nasa_moon_displacement_lola_ldem16_float_km.tif",
            "earth_night": "data/raw/earth_texture/earth_blackmarble_2016_3600x1800.jpg",
            "earth_day": "data/raw/earth_texture/earth_bluemarble_topobathy_200407_5400x2700.jpg",
        },
        "conventions": {
            "colorspace": "线性 sRGB(未 gamma)。渲染器自行 tone-map + gamma。",
            "moon_render": "月面食光=反照率纹理 × 食光颜色(线性相乘); 见 render_textured.py",
            "earth_render": "暗夜面(Black Marble自发光) + 体积大气环(径向查LUT,方位亮度调制)。",
        },
    }
    with open(os.path.join(OUTDIR, "meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    print("导出月食视频物理数据 →", OUTDIR)
    a, mrgb = export_moon_color_lut()
    print(f"  月面色 LUT: {len(a)} 点, 角距 0-60'")
    h, ergb = export_earth_ring_lut()
    print(f"  地球环色 LUT: {len(h)} 点, 擦边高度 0-{RE.H_ATM:.0f}km")
    Ds = export_frames()
    print(f"  帧几何: {len(Ds)} 帧, D 0-60'")
    export_meta()
    print("  meta.json ✓")
    print("完成。引擎无关，可喂 Unreal/Blender。")
