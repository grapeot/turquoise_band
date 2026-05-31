"""诊断：修正后的对侧-limb 角度映射，验证绿松石带贴本影边界。

矛盾根源(已解): r(h)=(R⊕+h)-α·d_moon 是太阳贴轴侧 limb 的光线落点, 描述红核对.
但绿松石带在月盘外缘, 由太阳对侧 limb 穿臭氧层的高 h 光照亮, 要用镜像公式:
  r_from_axis(h) = R_umbra - (α(h)·d_moon - h)
归一化基准用 R_umbra(4601km), 不是 R⊕.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import geometry as g

R_u = g.umbra_radius_km()
d = g.D_MOON_KM


def r_opposite_limb(h):
    """对侧-limb 公式: 绿松石带/月盘外缘用。"""
    alpha = g.refraction_angle(h)
    return R_u - (alpha * d - h)


def arcmin_opp(h):
    return np.degrees(np.arctan(np.abs(r_opposite_limb(h)) / d)) * 60


print(f"R_umbra = {R_u:.0f} km = {g.umbra_radius_arcmin():.1f} arcmin")
print(f"文献: 绿松石带(h~20-40km)应贴边界 41', 外移仅 0.78-1.9'\n")
print("对侧-limb 映射 (绿松石带用):")
print("h(km)  α(arcmin)  r距轴(km)  角距(arcmin)  位置")
for h in [0, 8, 18, 25, 30, 40, 55]:
    a = np.degrees(g.refraction_angle(h)) * 60
    r = r_opposite_limb(h)
    am = arcmin_opp(h)
    loc = "红核深处" if am < 20 else ("过渡" if am < 38 else "绿松石带(贴边界)")
    print(f"{h:>4}   {a:>7.2f}   {r:>7.0f}    {am:>6.1f}      {loc}")

# 自查: 绿松石带 h=20-40 应落在 ~38-42 arcmin
am40 = arcmin_opp(40)
am25 = arcmin_opp(25)
print(f"\nh=40km 角距={am40:.1f}' (应~41'), h=25km 角距={am25:.1f}'")
assert 36 < am40 < 43, "绿松石带(高h)应贴本影边界 ~41'"
assert am25 < am40, "h越高越贴边界"
print("修正验证通过: 绿松石带贴本影边界, 与文献自洽。")
