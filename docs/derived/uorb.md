# Wave Orbital Velocity — `uorb`

Significant horizontal wave orbital velocity at any height above the seabed (or depth below the surface), computed from integrated spectral parameters using the monochromatic peak-period approximation (Soulsby 1997).

---

## Governing equation

The horizontal orbital velocity at height *z* above the seabed in water of depth *h* is:

$$
U(z) = \frac{\pi H_s}{T} \cdot \frac{\cosh(k z)}{\sinh(k h)}
$$

where *k* is the angular wavenumber satisfying the linear dispersion relation ω² = g k tanh(kh) with ω = 2π/T.

At the seabed (z = 0, the default), cosh(0) = 1 and the formula reduces to the classical near-bed form used in sediment-transport and bed-shear-stress models:

$$
U_b = \frac{\pi H_s}{T \sinh(k h)}
$$

This is exact for a monochromatic wave of height Hs propagating in depth h (Soulsby 1997, eq. 2.44).

---

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `hs` | `"hs"` | Significant wave height variable name (m) |
| `tp` | `"tp"` | Representative wave period variable name (s) |
| `depth` | `"depth"` | Water depth variable name (m, positive downward) or scalar |
| `z` | `0.0` | Position in metres — variable name or scalar. Interpretation set by `reference`. |
| `reference` | `"bed"` | Reference datum for `z`: `"bed"` = height above seabed, `"surface"` = depth below surface |
| `g` | `9.81` | Gravitational acceleration (m/s²) |
| `solver` | `"explicit"` | Dispersion solver: `"explicit"` or `"exact"` |

---

## Depth profiles and reference datum

The `reference` parameter controls how `z` is interpreted:

- **`reference: 'bed'`** (default) — `z` is height above the seabed. `z = 0` is the seabed, the standard input for sediment-transport and bed-shear-stress models.
- **`reference: 'surface'`** — `z` is depth below the still-water surface. `z = 0` is the surface. Internally converted as `z_bed = depth − z`, so spatially varying bathymetry is handled automatically.

```yaml
calls:
  - func: mean
    data_vars: [uorb_bed, uorb_1m, uorb_10m]
    derived_vars:
      - name: uorb_bed
        func: uorb
        z: 0.0
        reference: bed        # seabed (default)
      - name: uorb_1m
        func: uorb
        z: 1.0
        reference: bed        # 1 m above bed
      - name: uorb_10m
        func: uorb
        z: 10.0
        reference: surface    # 10 m below the surface
```

The `reference: 'surface'` form is particularly convenient when you have a moored instrument or fixed platform at a known depth regardless of local bathymetry.

---

## Choice of representative period

The result is sensitive to the choice of *T*. Wave models typically output the peak period *T_p*. For improved accuracy, the mean zero-upcrossing period *T_z* = √(m₀/m₂) is preferred because it weights all spectral components more evenly. Conversion factors for common spectral shapes (Soulsby 1997, Table 2.1):

| Spectral shape | T_z / T_p |
|---|---|
| JONSWAP (γ = 3.3, typical wind sea) | ≈ 1/1.28 |
| Pierson–Moskowitz (fully developed) | ≈ 1/1.05 |

If only *T_p* is available, pre-compute *T_z* as a separate derived variable before calling `uorb`.

---

## Dispersion solvers

Two dispersion-relation solvers are available via the `solver` parameter:

**`'explicit'`** (default) — Chen & Thomson (1993) polynomial approximation:

$$
a = 1 + 0.6522\, k_0 h + 0.4622\, (k_0 h)^2 + 0.0864\, (k_0 h)^4 + 0.0675\, (k_0 h)^5
$$
$$
k = \frac{k_0 h \cdot \sqrt{1 + 1/(k_0 h \cdot a)}}{h}, \quad k_0 h = \omega^2 h / g
$$

Maximum error < 0.2 % across all depth regimes — negligible compared with Hs/Tp uncertainties (typically ≥ 5–10 %). Implemented as pure xarray/numpy ufuncs: no `apply_ufunc`, fully dask-transparent, ~40× fewer FLOPs than the iterative solver. Recommended for all production runs.

**`'exact'`** — Newton-Raphson iteration to full float64 machine precision. The Python loop is confined inside a numpy function dispatched by `apply_ufunc`, so the dask graph stays flat. Use for validation or sensitivity studies.

---

## Numerical stability

The transfer function cosh(kz)/sinh(kh) is evaluated in the equivalent form:

$$
\frac{e^{kz - kh} + e^{-kz - kh}}{1 - e^{-2kh}}
$$

This avoids float64 overflow for large kh (sinh overflows for kh ≳ 710) while remaining exact across all depth regimes. At large kh the numerator approaches exp(−k(h−z)), recovering the familiar exponential decay of orbital motion in deep water.

---

## Accuracy and limitations

This formula is exact for a single sinusoidal wave. For a random sea, Wiberg & Sherwood (2008) show the true RMS near-bed orbital velocity (from spectral integration) can differ from the monochromatic approximation by 10–30 %, depending on spectral shape and relative depth. When the full variance spectrum S(f) is available, the spectral integral formulation is preferred:

$$
\sigma_u^2(z) = \int_0^{\infty} (2\pi f)^2 \left[\frac{\cosh(k(f)\,z)}{\sinh(k(f)\,h)}\right]^2 S(f)\,\mathrm{d}f
$$

---

## References

- Soulsby, R. L. (1997). *Dynamics of Marine Sands*. Thomas Telford, London. — Near-bed orbital velocity (eq. 2.44) and period conversion factors (Table 2.1).
- Wiberg, P. L., & Sherwood, C. R. (2008). Calculating wave-generated bottom orbital velocities from surface-wave parameters. *Computers & Geosciences*, 34, 1243–1262. <https://doi.org/10.1016/j.cageo.2008.02.010>
- Nielsen, P. (1992). *Coastal Bottom Boundary Layers and Sediment Transport*. World Scientific. — Near-bed wave kinematics.
- Chen, G., & Thomson, J. (1993). A two-dimensional model of wave transformation in shallow water. *Ocean Engineering*, 20(6), 487–507. — Explicit polynomial approximation to the dispersion relation.
- Beji, S. (2013). Improved explicit approximation of linear dispersion relationship for gravity waves. *Coastal Engineering*, 73, 11–12. <https://doi.org/10.1016/j.coastaleng.2012.10.002>

---

## API reference

::: gridstats.derived.uorb.uorb
    options:
      show_source: false
      heading_level: 3
