"""【共享基础】颜色科学：透射光谱 → CIE XYZ → sRGB，并提取色相角与亮度。

用 colour-science 库提供的权威 CIE 1931 2° 色匹配函数和 sRGB 转换。
关键设计（见 RFC）：进入本影时绝对亮度暴跌，所以我们分两个层面看颜色——
  - hue（色相角）：反映"偏红/偏青/偏白"，是绿松石带的核心判据，对整体缩放不敏感。
  - luminance（亮度 Y）：反映真实消光导致的变暗，应保留相对量级。
"""
import numpy as np
import colour

# CIE 1931 2° 标准观察者色匹配函数
_CMF = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]

# sRGB(D65) XYZ→线性RGB 变换矩阵。单一来源, ray tracing 取线性 R/B 用。
M_XYZ2RGB_LINEAR = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])


def spectrum_to_XYZ(lam_nm, spec):
    """把谱 spec(lam) 积分成 CIE XYZ（未归一化，保留相对亮度）。

    参数
    ----
    lam_nm : 波长网格 (nm)
    spec   : 该网格上的辐射量（任意一致单位）
    返回 (X, Y, Z)。
    """
    lam_nm = np.asarray(lam_nm, dtype=float)
    spec = np.asarray(spec, dtype=float)
    # 把 CMF 插值到我们的波长网格
    xbar = np.interp(lam_nm, _CMF.wavelengths, _CMF.values[:, 0], left=0, right=0)
    ybar = np.interp(lam_nm, _CMF.wavelengths, _CMF.values[:, 1], left=0, right=0)
    zbar = np.interp(lam_nm, _CMF.wavelengths, _CMF.values[:, 2], left=0, right=0)
    dlam = np.gradient(lam_nm)
    X = np.sum(spec * xbar * dlam)
    Y = np.sum(spec * ybar * dlam)
    Z = np.sum(spec * zbar * dlam)
    return np.array([X, Y, Z])


def XYZ_to_sRGB(XYZ, normalize=None):
    """CIE XYZ → sRGB（0-1，已做 gamma 编码与 gamut 截断）。

    normalize: 若给定标量，先把 XYZ 除以它（用于把一组样本统一缩放到可显示范围，
    保留样本间的相对亮度差异）。
    """
    XYZ = np.asarray(XYZ, dtype=float)
    if normalize is not None and normalize > 0:
        XYZ = XYZ / normalize
    rgb = colour.XYZ_to_sRGB(XYZ)
    return np.clip(rgb, 0.0, 1.0)


def hue_angle(XYZ, white_XYZ=None):
    """从 XYZ 提取 CIE Lab 的色相角（度，0=红, 90=黄, 180=青绿, 270=蓝）。

    用 Lab 的 hue angle 比 RGB 更感知均匀。对绝对亮度缩放不敏感。
    white_XYZ：Lab 的参考白点（应为未衰减入射日光的 XYZ，与亮度归一用同一参考，
    见 science review）。缺省退化到对 XYZ 自身归一（仅用于模块自测）。
    """
    XYZ = np.asarray(XYZ, dtype=float)
    if XYZ[1] <= 0:
        return np.nan
    if white_XYZ is None:
        white_XYZ = XYZ / np.sum(XYZ)  # 自测兜底
    Lab = colour.XYZ_to_Lab(XYZ, illuminant=_xy_from_XYZ(white_XYZ))
    a, b = Lab[1], Lab[2]
    return np.degrees(np.arctan2(b, a)) % 360.0


def _xy_from_XYZ(XYZ):
    """XYZ → CIE xy 色度坐标（Lab 参考白点需要 xy）。"""
    XYZ = np.asarray(XYZ, dtype=float)
    s = np.sum(XYZ)
    if s <= 0:
        return np.array([0.3127, 0.3290])  # D65 兜底
    return np.array([XYZ[0] / s, XYZ[1] / s])


def luminance(XYZ):
    """相对亮度 Y。"""
    return float(np.asarray(XYZ, dtype=float)[1])


if __name__ == "__main__":
    # 自查：纯红、纯青绿、白的色相角应分别落在 红区、青绿区、低饱和
    lam = np.linspace(380, 780, 401)

    def gauss(c, w):
        return np.exp(-0.5 * ((lam - c) / w) ** 2)

    red = gauss(680, 20)
    teal = gauss(500, 30)  # 青绿
    white = np.ones_like(lam)

    for name, s in [("红 680nm", red), ("青绿 500nm", teal), ("白(平谱)", white)]:
        XYZ = spectrum_to_XYZ(lam, s)
        print(f"{name}: 色相角={hue_angle(XYZ):.1f}°, 亮度Y={luminance(XYZ):.3f}, sRGB={XYZ_to_sRGB(XYZ, normalize=luminance(XYZ)).round(2)}")
    print("自查：红的色相角应在 ~0-40°，青绿应在 ~150-200°，白应低饱和。")
