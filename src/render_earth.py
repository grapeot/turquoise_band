"""【渲染应用】从月面中心看地球：黑地球盘 + 物理大气折射环。

月食时从月球看是"日全食"——地球(角半径~57')完全盖住太阳(角半径~16')，
背光的黑地球盘外缘镶一圈被折射点亮的大气环。这圈环就是照亮月亮的"血月之光"源头。

大气环的两个维度（关键物理）：
- 径向（环内边→外边）= 擦边高度 0→~80km = 透射谱 红→青绿→白（复用 render_rt 的物理）
- 方位（绕环一圈 φ）= 那个方向太阳的可见量 = 亮度。月心距本影中心 D 决定哪侧亮：
  太阳中心相对地球中心偏移 D，朝太阳那侧的 limb 外露出太阳→最亮，背侧暗。
  D 增大→一侧越来越亮→ D>地球角半径-太阳角半径 时太阳缘探出 = 钻石环。

复用 render_rt 的辐射传输(擦边高度→透射谱颜色)与 render.py 显示链。
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import render_rt
import render as R
import geometry as g

OUT = R.OUT

# 角尺寸（arcmin），从月球看
ANG_EARTH = np.degrees(np.arctan(g.R_EARTH / g.D_MOON_KM)) * 60.0   # 地球角半径 ≈57'
ANG_SUN = np.degrees(np.arctan(g.R_SUN_KM / g.D_SUN_KM)) * 60.0     # 太阳角半径 ≈16'
H_ATM = 30.0                                                         # 环径向只映射 0-30km（红→青绿精华段，>30km 趋白不上环）
# 大气环真实角厚度仅 ~1.3%（H_ATM/R_E），细到几乎看不见。展示用夸张放大，
# 标注清楚这是艺术放大（径向颜色映射的物理不变，只是把薄环拉宽以可见）。
RING_FRAC_TRUE = 80.0 / g.R_EARTH                                    # ~1.3% 真实(0-80km全大气)
# 环厚度用**真实物理值**(1.3%)，不撑厚。要看清靠小 fov 长焦放大——这样曲率也跟着
# 放大(那一小段弧被放大显得更弯)，才是真实的"长焦看薄环"，而非把环physically撑厚。
RING_FRAC = RING_FRAC_TRUE                                          # ~1.3% 真实厚度


def _ring_color_table(n=400):
    """大气环径向(擦边高度0→H_ATM) → 线性 XYZ。复用 render_rt 的透射谱物理。

    擦边高度 h 的透射光颜色 = 我们一直在算的那个谱。这里直接取 LUT 的 XYZ_grid。
    """
    t = render_rt.build_branch_tables(n_h=n, h_max=H_ATM)
    # 按擦边高度排序（h_grid 已升序），返回 h→XYZ
    return t["h_grid"], t["XYZ_grid"]


def render_earth_frame(D_arcmin, size=900, ssaa=2, expose_srgb=0.85,
                       sun_dir_deg=0.0, earth_tex=None, ring_tables=None,
                       fov=None, center=None, return_linear=False, draw_sun=True):
    """渲染一帧"从月面中心看地球"。

    D_arcmin: 月心距本影中心角距（= 太阳中心相对地球中心的偏移），0=正中本影。
    sun_dir_deg: 太阳偏移的方位（默认 0=朝 +x/右；决定环哪侧亮）。
    fov: 视场半宽(arcmin)。None=看整个地球(默认~1.15·地球半径)；给小值=长焦只看一段弧。
    center: (cx,cy) 画幅中心(arcmin)，长焦时对准要看的那段环。
    返回 rgb8 (size,size,3)。
    """
    if ring_tables is None:
        ring_tables = _ring_color_table()
    h_grid, XYZ_grid = ring_tables

    # 画幅：默认看整个地球；fov 给定则长焦放大到那段
    half = fov if fov is not None else ANG_EARTH * 1.15
    cx, cy = center if center is not None else (0.0, 0.0)
    S = size * ssaa
    xs = np.linspace(cx - half, cx + half, S)
    ys = np.linspace(cy + half, cy - half, S)   # 行0=高y(图顶), 让世界y朝上=图上(修方向翻转)
    X, Y = np.meshgrid(xs, ys)
    r = np.hypot(X, Y)                       # 距地球中心角距 (arcmin)
    phi = np.arctan2(Y, X)                   # 方位角

    rgb = np.zeros((S, S, 3))

    # --- 1. 地球黑盘（夜面剪影 + 可选真实纹理压暗）---
    disk = r <= ANG_EARTH
    if earth_tex is not None:
        night = _sample_earth(earth_tex, X, Y, disk, sun_dir_deg, D_arcmin)
        rgb[disk] = night[disk]
    # 否则保持黑

    # --- 2. 大气折射环 ---
    # 环在 r ∈ [ANG_EARTH, ANG_EARTH*(1+RING_FRAC)]，径向映射擦边高度 0→H_ATM
    ring_in = ANG_EARTH
    ring_out = ANG_EARTH * (1.0 + RING_FRAC)
    in_ring = (r >= ring_in) & (r <= ring_out)
    # 径向归一 → 擦边高度（内边r=ANG_EARTH是h=0低空, 外边是h=H_ATM高空）
    t_rad = np.clip((r - ring_in) / (ring_out - ring_in), 0, 1)
    h_pix = t_rad * H_ATM
    # 查透射谱颜色
    XYZ_ring = _interp_xyz(h_pix, h_grid, XYZ_grid)

    # 方位亮度：太阳中心在地球后方偏移 D，朝 sun_dir 方向。
    # 环上方位 φ 处，limb 外缘到太阳中心的角距 = |该方位地球缘点 - 太阳中心|。
    # 简化：太阳照亮量 ∝ 该方位朝向太阳的程度 + 太阳是否被地球完全挡。
    sun_phi = np.radians(sun_dir_deg)
    # 该方位 limb 点的位置（地球缘）相对太阳中心的角距
    # 方位亮度物理：环上方位 φ 处的折射光，来自太阳圆盘在该方位 limb 外的"贴近程度"。
    # 太阳中心相对地球中心偏移 D（朝 sun_phi 方向）。该方位 limb 到太阳中心的角距：
    lx = ANG_EARTH * np.cos(phi); ly = ANG_EARTH * np.sin(phi)
    sx = D_arcmin * np.cos(sun_phi); sy = D_arcmin * np.sin(sun_phi)
    dist_to_sun = np.hypot(lx - sx, ly - sy)     # limb点到太阳中心角距
    # 该方位 limb 的折射光强 ∝ 太阳在该方位 limb **外侧的贴近程度**。
    # 太阳贴近该 limb 点(dist_to_sun 小)→光从该方位强折射进来→亮。
    #   D=0: 太阳正中, 各 φ 的 dist 都=ANG_EARTH(均匀)→ 全圈日落环。
    #   D 大: 朝 sun_phi 那侧 dist 变小(太阳贴近该缘)→越来越亮; 背侧 dist 变大→暗。
    #   单调: dist 越小越亮(用 1/(1+dist) 型), 太阳露出侧随 D 增持续变亮(修复:原公式峰值
    #   错设在 dist=ANG_EARTH, 导致太阳贴近时反而变暗)。
    soft = ANG_SUN * 3.0
    bright = 0.05 + 0.9 * np.exp(-(dist_to_sun / soft) ** 2)
    # 钻石环：太阳边缘探出地球缘(D+ANG_SUN > ANG_EARTH)时，朝 sun_phi 那侧出现强光点
    sun_peek = D_arcmin + ANG_SUN - ANG_EARTH    # >0 即露出
    if sun_peek > 0:
        # 露出方向(朝 sun_phi)的 limb 强光，按方位接近 sun_phi 程度
        align = np.cos(phi - sun_phi)            # 1=朝太阳侧
        spike = np.clip((align - 0.6) / 0.4, 0, 1) * np.clip(sun_peek / ANG_SUN, 0, 1)
        bright = bright + spike * 5.0            # 钻石环强光

    # 颜色用各擦边高度的色相（按自身亮度归一，让红→青绿→白的色相显出来），
    # 亮度由方位(太阳可见量)调制——这样既看得见日落环的颜色梯度，又有 D 决定的一侧亮。
    Yr = np.maximum(XYZ_ring[..., 1], 1e-9)
    chroma = XYZ_ring / Yr[..., None]               # 色相(Y=1)
    rgb_ring = R._xyz_to_srgb_linear(chroma)
    rgb_ring = np.clip(rgb_ring, 0, None)
    rgb_ring = rgb_ring / np.maximum(rgb_ring.max(axis=-1, keepdims=True), 1e-6)  # 饱和归一
    rgb_ring = rgb_ring * bright[..., None]         # 方位亮度调制
    rgb[in_ring] = rgb_ring[in_ring]

    # === 太阳：露出地球缘的那部分(钻石环的"钻石") ===
    # 太阳盘(中心 sx,sy, 半径 ANG_SUN)落在地球外(r>地球缘)的部分=可见太阳新月。
    # 只在**全景**(draw_sun=True)画；特写(长焦)不画以免铺满。亮度不求真实(够亮即可)。
    r_to_sun = np.hypot(X - sx, Y - sy)
    sun_visible = (r_to_sun <= ANG_SUN) & (r > ANG_EARTH * (1.0 + RING_FRAC))
    if draw_sun and np.any(sun_visible):
        # 太阳盘内越靠中心越亮(暖白强光)
        edge = np.clip((ANG_SUN - r_to_sun) / ANG_SUN, 0, 1)
        sun_rgb = np.stack([np.full_like(X, 1.0), np.full_like(X, 0.93),
                            np.full_like(X, 0.80)], axis=-1)
        sun_lin = sun_rgb * (2.0 + 6.0 * edge[..., None])
        rgb[sun_visible] = sun_lin[sun_visible]

    if return_linear:
        return _box(np.clip(rgb, 0, None), ssaa)        # 线性HDR(供16bit/后期)
    # tone map + gamma（环很亮，地球很暗）
    rgb = R._srgb_gamma(np.clip(rgb, 0, 1))
    rgb8 = (np.clip(_box(rgb, ssaa), 0, 1) * 255 + 0.5).astype(np.uint8)
    return rgb8


def _ring_peak(XYZ_grid):
    return float(np.max(XYZ_grid[:, 1]))


def _interp_xyz(h_pix, h_grid, XYZ_grid):
    idx = np.clip(np.searchsorted(h_grid, h_pix), 1, len(h_grid) - 1)
    h0, h1 = h_grid[idx - 1], h_grid[idx]
    w = ((h_pix - h0) / np.maximum(h1 - h0, 1e-9))[..., None]
    return (1 - w) * XYZ_grid[idx - 1] + w * XYZ_grid[idx]


def _box(img, f):
    if f == 1:
        return img
    s = img.shape[0]; n = s // f
    return img[:n*f, :n*f].reshape(n, f, n, f, img.shape[2]).mean(axis=(1, 3))


def _sample_earth(tex, X, Y, disk, sun_dir_deg, D_arcmin):
    """把地球纹理正交投影贴到盘上，压暗成夜面剪影（背光面朝月球）。"""
    out = np.zeros(X.shape + (3,))
    u = X / ANG_EARTH; v = Y / ANG_EARTH
    z = np.sqrt(np.clip(1 - u*u - v*v, 0, 1))
    lat = np.arcsin(np.clip(v, -1, 1))
    lon = np.arctan2(u, z)
    H, W = tex.shape[:2]
    ci = np.clip(((np.degrees(lon) + 180) / 360 * (W-1)).astype(int), 0, W-1)
    ri = np.clip(((90 - np.degrees(lat)) / 180 * (H-1)).astype(int), 0, H-1)
    col = tex[ri, ci]
    # 夜面：日食时朝月球的是地球背光面。用 Black Marble 城市灯光图——城市亮点保留，
    # 海陆夜面极暗。灯光本身已是"暗背景+亮点"，乘一个适中系数即可（不压太狠，让灯光可见）。
    out[disk] = np.clip(col[disk] * 0.0225, 0, None) # 夜面压暗到原一半(地球本身更暗,环独立不受影响)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--D", type=float, default=20.0)
    ap.add_argument("--size", type=int, default=800)
    ap.add_argument("--fov", type=float, default=None, help="长焦视场半宽(arcmin)，小=放大看一段弧")
    ap.add_argument("--cx", type=float, default=0.0)
    ap.add_argument("--cy", type=float, default=None, help="画幅中心y(默认对准底部环)")
    args = ap.parse_args()
    print(f"地球角半径 {ANG_EARTH:.0f}', 太阳角半径 {ANG_SUN:.0f}', 环厚 {RING_FRAC*100:.0f}%(放大~8×)")
    rt = _ring_color_table()
    # 地球纹理
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    texp = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "earth_texture",
                        "earth_blackmarble_2016_3600x1800.jpg")
    etex = np.asarray(Image.open(texp).convert("RGB"), float) / 255.0 if os.path.exists(texp) else None
    # 长焦默认对准底部那段环(cy=-地球半径附近)
    cy = args.cy if args.cy is not None else (-ANG_EARTH if args.fov else 0.0)
    center = (args.cx, cy) if args.fov else None
    # 太阳朝下(-90°)，亮侧/钻石环在底部，长焦看底部即看亮侧(=绿松石带取光的那侧)
    rgb8 = render_earth_frame(args.D, size=args.size, ring_tables=rt, earth_tex=etex,
                              fov=args.fov, center=center, sun_dir_deg=-90.0)
    from PIL import Image
    p = os.path.join(OUT, f"earth_D{args.D:.0f}.png")
    Image.fromarray(rgb8).save(p)
    print(f"已存 {p}")
