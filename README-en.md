# turquoise_band — Recreating the turquoise band of a total lunar eclipse with code

During a total lunar eclipse, the Moon doesn't go completely black inside Earth's shadow. Instead, it takes on a sequence of colors: deep red in the heart of the umbra, then a narrow teal-green band (the turquoise band), and farther out, back to the normal moonlit white. This project builds a model from the fundamental physics of light traveling through the atmosphere, computes those colors quantitatively, and ultimately synthesizes an eclipse photo that matches reality.

**Site (images, videos, and six-step walkthrough)**: https://grapeot.github.io/turquoise_band/ — For a deeper dive into the physics, see the principles page on the site.

The physical intuition goes like this: when sunlight refracts through Earth's atmosphere toward the eclipsed Moon, each ray grazes the planet at a different altitude — some skim the lower atmosphere, some pass through the stratosphere, and some cut across much higher up. These altitude differences determine the light's color.

In the lower atmosphere, air molecules scatter short-wavelength (blue) light far more strongly than long-wavelength (red) light. This is Rayleigh scattering — the same effect that makes the sky blue and sunsets red. After the blue gets scattered away, the remaining red light lands on the Moon, giving us the blood moon. Up at stratospheric altitudes, ozone has a strong absorption band between 500–700 nm (called the Chappuis band) that eats away orange and red light. Once the reds are removed, the light that gets through appears teal-green — that is where the turquoise band comes from. Higher still, the atmosphere is thin, absorption and scattering are minimal, and the light settles close to ordinary daylight white.

## Status

- [x] **L0 Radiative transfer closed loop**: hue curve red→teal→white, with measured data (O₃ Serdyuchenko 2014, AFGL atmosphere, SAO2010 solar spectrum, CIE CMF)
- [x] **L1 Refraction geometry (analytic)**: single mapping `r(h)=(R⊕+h)−α·d_moon`, matching Mallama Table 3.1 within <2% at mid-to-high h (~6-10% at the low-h end)
- [x] **L2 Per-pixel reverse ray-tracing render** (render_rt.py): branch-aware lookup, no banding, sub-second via numpy
- [x] **L3 Photorealistic moon disk**: NASA lunar texture × physics-based eclipse colors, log tone map, faithful brightness (the blue band really is that narrow physically)
- [x] **Solar-disk ray tracing** (brute_ray_trace.py): the 32' solar disk is 5× wider than the band, softening the point-source model's over-saturated deep teal (0.41) to an observationally plausible pale teal (0.70)
- [x] **True forward ray tracing** (raytrace_eclipse.py, the current authoritative pipeline): true refraction via RK4 + curved-path extinction (using actual tangent heights) + emergent focusing from ray casting + penumbral direct light, zero analytic prescriptions. Umbra center ≈−13.5 stops
- [x] **HDR gain map**: compatible with iPhone Photos HEIC (tools/make_gainmap_hdr.swift)

### Core physical conclusions

The turquoise band is intrinsically teal (cyan-green), faint, and narrow. Along a radial slice of the moon disk, the color (red/blue ratio, R/B) shifts smoothly, but the visually apparent width is small — and that is a brightness effect. The bluest point is about 10 stops dimmer than a full Moon, while the adjacent near-white zone is much brighter, so only the thinnest sliver closest to the blue band stands out to the eye. The current model yields a minimum R/B of roughly 0.70 (at 41' from disk center, raw sRGB measurement standard). Converted to Shu (2024)'s 654 nm/491 nm satellite narrow-band standard, the in-band R/B falls between 0.86 and 1.0, overlapping Shu's measured range of 0.8–1.0. (These two measurement standards are not directly comparable; see working.md continuation 16 and the 2026-06-09 corrigendum.) Those "half-disk saturated blue" eclipse images you see online are post-processing artifacts.

The umbra center appearing darker than expected has been traced to a calibration discrepancy in the extinction standard; see working.md 2026-06-09/06-10 for the analysis.

### Scientific scope

The color and brightness of eclipse light are determined by atmospheric radiative transfer. How it looks on your screen is determined by exposure and tone mapping — two independent systems. This project nearly misattributed screen effects to physics on two occasions (see working.md Lessons). The model's scope:

- The umbra-center value of −15.1 stops corresponds to the clearest-sky, aerosol-free conditions. A purely molecular atmosphere (no aerosols at all) has a theoretical ceiling of roughly −13.5 stops. A real eclipse can be much darker if stratospheric aerosols — volcanic ash, for instance — are present at the time. **This model cannot tell you whether the next eclipse will be bright or dark; that depends on what stratospheric aerosols are doing at the time.**
- The Shu (2024) satellite narrow-band R/B < 1 ribbon cannot be reproduced under one-dimensional radial averaging, due to red-shoulder extinction and measurement-standard differences. This remains an open item; see docs/MODEL_CARD.md.
- The I band lacks water-vapor and oxygen absorption data; absolute brightness values in that band are not cited.

### Key outputs (outputs/)

- `moon_disk_turquoise_final.png` — photorealistic eclipse photo (disk physics, d=40, bluest point passes through disk center)
- `moon_composite.png` — cross-section diagnostic (quantifying blue-band width)
- `moon_brightness_cliff.png` — two-exposure comparison (showing that "the blue band looks narrow because of brightness")
- `ablation/step_*.tif` — six-step mechanism ablation (plain disk → occlusion → Rayleigh → ozone → solar disk → true ray tracing)
- `moon_eclipse_sdr/hdr_h265.mp4` — dual-perspective eclipse video (lunar disk | Earth panorama | atmospheric ring close-up | spectral origin)

### Run

```bash
source .venv/bin/activate
python src/raytrace_eclipse.py         # True forward ray tracing (authoritative pipeline, ~3s)
python -m pytest src/tests/ -v         # Physical invariants + render smoke tests (incl. slow, ~2min)
python src/render_textured.py          # Photorealistic moon disk (default d=40, authoritative ray tracing LUT; --engine pointsource is historical comparison only)
python src/diag_composite.py           # Blue-band width diagnostic
python src/pipeline.py --self-check    # L0 hue curve
bash scripts/make_hdr.sh               # Generate HDR HEIC (requires HDR_REFERENCE_HEIC pointing to any iPhone-captured HEIC)
```

## Docs

- `docs/working.md` — **Current authoritative record**: Changelog + Key Technical Decisions + Lessons (read this first)
- `docs/PHYSICS_AND_PITFALLS.md` — Physics methods + full pitfall log
- `docs/L1_geometry.md` — Refraction geometry prescription (single mapping, verified against Mallama)
- `docs/PRD.md` / `docs/RFC.md` — Goals / technical design
- `docs/SCIENCE_REVIEW.md` — Literature survey summary
- `docs/LOG.md` — Early log (archived; some conclusions superseded by working.md)

## Environment

```bash
source .venv/bin/activate   # uv venv, Python 3.12
```

Machine: Apple M3 Ultra (32 CPU / 80 GPU / 512GB). Full pipeline is numpy-vectorized on CPU (full ray tracing ~3s); MPS not used.
