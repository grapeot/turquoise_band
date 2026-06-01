"""Reproduce the literature (Shu et al. 2024, RemoteSens 16(22):4181) turquoise-band
ribbon-width measurement geometry, on OUR physics.

Literature definition (verified):
  - R/B = B4(654nm)/B2(491nm) radiance ratio, computed per-pixel on a 2D GF-4 lunar-disk image.
  - The "turquoise ribbon" is the boundary transition strip at the penumbra-umbra edge where
    R/B crosses from >1 (reddish umbra interior) down to <1 (teal), NOT the entire R/B<1 range.
  - Reported radial full width of that boundary ribbon: 120-190 km on the lunar surface
    (~1.1-1.7 arcmin at 112 km/arcmin), with in-band R/B ~ 0.8-1.0.

Why a naive "R/B<1 interval along radius" is wrong here:
  In our anti-solar-axis single-ray / disk model the surface keeps getting bluer outward
  (B2 grows monotonically as the limb height h increases), so R/B<1 spans ~24' (~2700 km).
  The literature ribbon is bounded OUTWARD by the brightness cliff: crossing the umbra edge
  into the penumbra, an ever-larger fraction of the (white) solar disk becomes directly
  visible, washing the blue back out (R/B -> 1) and saturating brightness. The teal is only
  visible as a distinct ribbon in the narrow window where R/B<1 AND the surface is bright
  enough to measure but not yet white-saturated.

This module:
  1. Builds band radiances B4(654), B2(491) and total brightness Y as functions of h.
  2. Integrates the finite solar disk (point_source toggle) over each lunar-surface angular
     distance a, INCLUDING the penumbra wash-to-white (subpoints whose refracted landing
     point leaves the traced atmosphere = direct white sunlight). This reproduces both the
     inward red->teal crossing and the outward teal->white return.
  3. Renders the 2D R/B image (umbra boundary crossing the disk) the way the literature does.
  4. Measures the ribbon radial full width as the strip where R/B<1 AND brightness is in the
     measurable, non-white-saturated window (the literature's visible-ribbon criterion).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import geometry as g
import radiative_transfer as rt
import solar
import render_rt as rrt

KM_PER_ARCMIN = 112.0          # lunar surface scale (task spec)
LAM = np.linspace(380.0, 780.0, 401)
J654 = int(np.argmin(np.abs(LAM - 654.0)))
J491 = int(np.argmin(np.abs(LAM - 491.0)))


def build_band_tables(n_h=8000, h_min=0.0, h_max=80.0):
    """h -> (a_signed, B4, B2, Y_band, focusing) plus direct-sunlight white band radiances.

    B4/B2 carry focusing (energy-conserving convergence) so the brightness cliff is physical.
    The 'white' band radiances are undeviated direct sunlight (penumbra / outside umbra),
    used for solar subpoints whose refracted landing point leaves the traced atmosphere.
    """
    h = np.linspace(h_min, h_max, n_h)
    I = rt.emergent_spectrum(h, LAM)                       # (H, L) emergent spectra vs h
    # geometric focusing (no r_floor; disk integration regularizes the center)
    r_signed_km = (g.R_EARTH + h) - g.refraction_angle(h) * g.D_MOON_KM
    a_signed = np.degrees(np.arctan(r_signed_km / g.D_MOON_KM)) * 60.0    # monotonic rising
    b = g.R_EARTH + h
    dr_dh = 1.0 + g.refraction_angle(h) * g.D_MOON_KM / g.H_REFRAC_KM
    foc = b / dr_dh / np.maximum(np.abs(r_signed_km), 1.0)

    B4 = I[:, J654] * foc
    B2 = I[:, J491] * foc
    # total photometric brightness Y (CIE) vs h, with focusing, for the saturation criterion
    import color as col
    Isun = solar.solar_spectrum(LAM)
    white_XYZ = col.spectrum_to_XYZ(LAM, Isun); kY = 1.0 / white_XYZ[1]
    Yb = np.array([col.spectrum_to_XYZ(LAM, I[i])[1] for i in range(len(h))]) * kY * foc

    white_B4 = float(Isun[J654])     # direct sunlight band radiances (penumbra / full sun)
    white_B2 = float(Isun[J491])
    white_Y = 1.0                    # direct moonlight normalized to Y=1
    # CRITICAL: the raw radiance ratio of the SOLAR spectrum itself is B4(654)/B2(491)=0.737<1
    # (the Sun is brighter at 491 than 654). So a raw-radiance R/B has white sunlight already
    # below the R/B=1 threshold and never returns above 1 -> no bounded ribbon. The literature
    # R/B is a calibrated/reflectance ratio where neutral white = 1; we reproduce that by
    # dividing out the solar slope (sun_rb). After this, white sunlight -> R/B=1 exactly and the
    # teal dip becomes a proper bounded ribbon (red>1 inside, teal<1, white->1 outside).
    sun_rb = white_B4 / white_B2     # ~0.737
    return dict(h=h, a_signed=a_signed, B4=B4, B2=B2, Yb=Yb, foc=foc,
                ang_sun=float(np.degrees(np.arctan(g.R_SUN_KM / g.D_SUN_KM)) * 60.0),
                white_B4=white_B4, white_B2=white_B2, white_Y=white_Y, sun_rb=sun_rb,
                a_lo=float(a_signed.min()), a_hi=float(a_signed.max()))


def integrate_disk(a_pixel, t, point_source=True, n_xi=257):
    """Integrate the solar disk over lunar-surface angular distance(s) a.

    Returns (B4, B2, Y) band radiances and brightness at each a.
    Solar subpoints xi in [-ang_sun, ang_sun] (chord-weighted, uniform disk). A subpoint's
    refracted light must land at a_signed(h) = a - xi -> invert for h -> band radiance.
      landing > a_hi : that subpoint's ray leaves the traced atmosphere = DIRECT WHITE SUN
                        (this is the penumbra: increasing white fraction across the edge).
      landing < a_lo : ray blocked by solid Earth -> 0.
    point_source=True : single axial subpoint (literature comparison baseline per task).
    """
    a = np.atleast_1d(np.asarray(a_pixel, float))[:, None]              # (P,1)
    if point_source:
        xi = np.array([[0.0]]); w = np.array([1.0])
    else:
        xi = np.linspace(-t["ang_sun"], t["ang_sun"], n_xi)[None, :]    # (1,X)
        w = np.sqrt(np.clip(t["ang_sun"] ** 2 - xi ** 2, 0, None)).ravel()
        w = w / max(w.sum(), 1e-12)
    target = a - xi                                                     # (P,X) required landing
    a_sig = t["a_signed"]
    tgt = np.clip(target, a_sig[0], a_sig[-1])
    idx = np.searchsorted(a_sig, tgt).clip(1, len(a_sig) - 1)
    a0 = a_sig[idx - 1]; a1 = a_sig[idx]; fr = (tgt - a0) / np.maximum(a1 - a0, 1e-9)

    def lerp(arr):
        return arr[idx - 1] * (1 - fr) + arr[idx] * fr
    b4 = lerp(t["B4"]); b2 = lerp(t["B2"]); yy = lerp(t["Yb"])
    over = target > t["a_hi"]      # leaves atmosphere -> direct white sunlight (penumbra)
    under = target < t["a_lo"]     # blocked by Earth
    b4 = np.where(over, t["white_B4"], b4); b4 = np.where(under, 0.0, b4)
    b2 = np.where(over, t["white_B2"], b2); b2 = np.where(under, 0.0, b2)
    yy = np.where(over, t["white_Y"], yy);  yy = np.where(under, 0.0, yy)
    B4 = (b4 * w).sum(1); B2 = (b2 * w).sum(1); Y = (yy * w).sum(1)
    return B4, B2, Y


def radial_profile(t, point_source=True, a_lo=38.0, a_hi=85.0, n=4000, n_xi=257,
                   sun_normalize=True):
    a = np.linspace(a_lo, a_hi, n)
    B4, B2, Y = integrate_disk(a, t, point_source=point_source, n_xi=n_xi)
    RB = B4 / np.maximum(B2, 1e-30)
    if sun_normalize:
        RB = RB / t["sun_rb"]        # reflectance-style: neutral white sunlight -> R/B = 1
    return a, RB, Y, B4, B2


def measure_ribbon(a, RB, Y=None, white_Y=1.0, y_floor_frac=0.02):
    """Measure the boundary-ribbon radial full width (the literature geometry).

    On the sun-normalized R/B (white sunlight = 1), the teal ribbon is the bounded strip
    R/B < 1 around the boundary, bracketed by two R/B=1 crossings:
      inner_edge = R/B drops below 1 (reddish umbra interior -> teal)
      outer_edge = R/B returns above 1 (teal -> white penumbra/full sun)
    This is exactly the literature's "boundary transition ribbon": it does NOT include the
    deep red umbra (R/B>1) nor the far white penumbra (R/B->1). Width = outer-inner -> km.

    y_floor_frac: only count the strip where surface brightness is above the dark umbra floor
    (so unphysically deep, unmeasurable umbra pixels don't get counted as ribbon).
    """
    yfrac = (Y / max(white_Y, 1e-30)) if Y is not None else np.ones_like(a)
    # locate the teal dip minimum (R/B<1) within the measurable-brightness band
    cand = (RB < 1.0) & (yfrac >= y_floor_frac)
    if not cand.any():
        return None
    imin = int(np.nanargmin(np.where(cand, RB, np.inf)))
    # inner R/B=1 crossing: scan inward (decreasing a) from the minimum
    inner = a[0]
    for i in range(imin, 0, -1):
        if (RB[i - 1] - 1) * (RB[i] - 1) <= 0:
            f = (1 - RB[i - 1]) / (RB[i] - RB[i - 1] + 1e-30)
            inner = a[i - 1] + f * (a[i] - a[i - 1]); break
    # outer R/B=1 crossing: scan outward (increasing a) from the minimum
    outer = a[-1]
    for i in range(imin, len(a) - 1):
        if (RB[i] - 1) * (RB[i + 1] - 1) <= 0:
            f = (1 - RB[i]) / (RB[i + 1] - RB[i] + 1e-30)
            outer = a[i] + f * (a[i + 1] - a[i]); break
    width_arcmin = abs(outer - inner)
    strip = (a >= inner) & (a <= outer)
    rb_in = RB[strip]
    return dict(inner=inner, outer=outer, width_arcmin=width_arcmin,
                width_km=width_arcmin * KM_PER_ARCMIN,
                rb_min=float(np.nanmin(rb_in)), rb_max=float(np.nanmax(rb_in)),
                a_bluest=float(a[imin]), rb_bluest=float(RB[imin]))


def render_rb_image(t, d_arcmin=55.0, size=600, point_source=True, n_xi=129):
    """2D R/B(654/491) image with the umbra boundary crossing the lunar disk (literature setup).

    Geometry same caliber as render_textured: anti-solar center at origin, lunar disk center
    at +x distance d, lunar radius R_MOON (~15.5'), umbra radius R_UMBRA (~41.2').
    d chosen so the umbra-penumbra boundary (and the teal ribbon at ~46-58') crosses the disk.
    """
    Rm = rrt.R_MOON_ARCMIN
    half = Rm + 1.0
    cx = d_arcmin
    xs = np.linspace(cx - half, cx + half, size)
    ys = np.linspace(-half, half, size)
    Xw, Yw = np.meshgrid(xs, ys)
    a = np.hypot(Xw, Yw)
    inside = np.hypot(Xw - cx, Yw) <= Rm
    flat = a[inside]
    B4, B2, Y = integrate_disk(flat, t, point_source=point_source, n_xi=n_xi)
    RBimg = np.full(a.shape, np.nan)
    Yimg = np.full(a.shape, np.nan)
    RBimg[inside] = (B4 / np.maximum(B2, 1e-30)) / t["sun_rb"]   # sun-normalized (white=1)
    Yimg[inside] = Y
    return dict(RB=RBimg, Y=Yimg, a=a, inside=inside, cx=cx, Rm=Rm,
                extent=(xs[0], xs[-1], ys[0], ys[-1]))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--disk", action="store_true", help="finite solar disk (default point source)")
    ap.add_argument("--raw", action="store_true", help="raw radiance R/B (no sun-normalization)")
    ap.add_argument("--y_floor", type=float, default=0.02, help="umbra-floor brightness frac")
    ap.add_argument("--d", type=float, default=55.0)
    args = ap.parse_args()
    ps = not args.disk
    sn = not args.raw

    t = build_band_tables(n_h=8000)
    print(f"R_umbra = {g.umbra_radius_arcmin():.2f} arcmin,  ang_sun = {t['ang_sun']:.2f} arcmin")
    print(f"solar B4/B2 (raw white R/B) = {t['sun_rb']:.4f}  -> sun_normalize={sn}")

    a, RB, Y, B4, B2 = radial_profile(t, point_source=ps, sun_normalize=sn)
    res = measure_ribbon(a, RB, Y, white_Y=t["white_Y"], y_floor_frac=args.y_floor)
    print(f"\n=== 1D radial profile (point_source={ps}, sun_normalize={sn}) ===")
    print("a(')   R/B     Y/white")
    for ai in [44, 46, 48, 49, 50, 51, 52, 54, 56, 57, 58, 60, 64]:
        k = int(np.argmin(np.abs(a - ai)))
        print(f" {a[k]:5.1f}  {RB[k]:6.3f}   {Y[k]/t['white_Y']:.3f}")
    if res:
        print(f"\nRibbon inner edge (R/B=1, red->teal)  : {res['inner']:.2f}'")
        print(f"Ribbon outer edge (R/B=1, teal->white): {res['outer']:.2f}'")
        print(f"Ribbon radial FULL WIDTH : {res['width_arcmin']:.2f} arcmin "
              f"= {res['width_km']:.0f} km")
        print(f"in-band R/B : {res['rb_min']:.3f} .. {res['rb_max']:.3f}  "
              f"(bluest R/B={res['rb_bluest']:.3f} @ {res['a_bluest']:.2f}')")
    print(f"\nLiterature: 120-190 km (~1.1-1.7'), in-band R/B 0.8-1.0")
