"""写实月食静态图（真·正向 ray tracing LUT 版）：d=40' 绿松石带横跨月盘。

旧版写实图（moon_disk_turquoise_final.png，working.md 续11）用的是金标准圆盘 LUT
（brute_ray_trace，本影偏亮 −7.7 档）。本脚本换成权威物理管线的 LUT
（raytrace_eclipse.build_lut_from_raytrace：真折射 RK4 + z_tan 弯曲消光 + 撒线涌现
+ 半影直射光 + 气溶胶两组分 + 太阳 limb darkening，与视频/ablation step6 同一套参数）。

纹理/曝光/tone map 管线完整沿用 render_textured.render_realistic_disk（NASA CGI Moon Kit
纹理、faithful 曝光、不自创 tone mapping）；动态范围压缩与饱和度沿用续11 旧版写实图的
记录参数（dyn_gamma=0.55、saturation=1.3——新 LUT 本影 −15 档，dyn_gamma=1.0 会让左半
死黑无层次）。不重构 render_textured，用模块级替换把 render_rt 的解析着色换成 RT LUT
着色（render_realistic_disk 只通过 build_branch_tables/shade 两个入口取物理颜色）。

用法: ./.venv/bin/python scripts/render_disk_rt.py [--d 40]
输出: outputs/moon_disk_turquoise_rt.png
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import raytrace_eclipse as rte
import render_rt
import render as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=float, default=40.0,
                    help="月心到本影中心角距 arcmin（默认 40，月盘横跨本影边界 41.2'）")
    ap.add_argument("--size", type=int, default=1400)
    ap.add_argument("--ssaa", type=int, default=2)
    ap.add_argument("--dyn-gamma", type=float, default=0.55,
                    help="物理亮度幂律压缩（续11 写实图记录值 0.55）")
    ap.add_argument("--saturation", type=float, default=1.3,
                    help="轻饱和（续11 写实图记录值 1.3）")
    args = ap.parse_args()

    print("建真 ray tracing LUT（与视频/ablation step6 同参数）...")
    t0 = time.time()
    lut = rte.build_lut_from_raytrace(n_rays_b=4_000_000, n_sun=2000,
                                      n_h_nodes=500, n_pix=300, n_disp=12)
    print(f"  LUT 完成 {time.time() - t0:.0f}s  (a 覆盖 {lut['a'][0]:.1f}'–{lut['a'][-1]:.1f}')")

    # 模块级替换：render_realistic_disk 取物理颜色的两个入口换成 RT LUT。
    # （render_textured 不重构；faithful 曝光路径只用 shade 的返回值，不碰 tables 内部。）
    render_rt.build_branch_tables = lambda **kw: lut
    render_rt.shade = lambda a, tables: render_rt.shade_disk_lut(a, lut)

    import render_textured as rtx
    t0 = time.time()
    rgb8, info = rtx.render_realistic_disk(d_arcmin=args.d, size=args.size, ssaa=args.ssaa,
                                           dyn_gamma=args.dyn_gamma,
                                           saturation=args.saturation)
    print(f"渲染 {args.size}x{args.size} (SSAA×{args.ssaa}) 用时 {time.time() - t0:.1f}s")

    out_path = os.path.join(R.OUT, "moon_disk_turquoise_rt.png")
    rtx.save_png_raw(rgb8, out_path)


if __name__ == "__main__":
    main()
