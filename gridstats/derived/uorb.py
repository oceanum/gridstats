"""Wave orbital velocity derived variable."""
from __future__ import annotations

import numpy as np
import xarray as xr

from gridstats.registry import register_derived


def _wavenumber(
    omega: xr.DataArray,
    h: xr.DataArray,
    g: float = 9.81,
) -> xr.DataArray:
    """Explicit wavenumber from the Chen & Thomson (1993) approximation.

    Computes the angular wavenumber *k* (rad/m) satisfying the linear
    dispersion relation ω² = g k tanh(kh) in a single vectorised pass —
    no iteration, no ``apply_ufunc``, fully dask-transparent.

    The formula is a polynomial approximation in the dimensionless deep-water
    parameter k₀h = ω²h/g:

    .. code-block:: none

        a   = 1 + 0.6522 k₀h + 0.4622 (k₀h)² + 0.0864 (k₀h)⁴ + 0.0675 (k₀h)⁵
        k   = k₀h · √(1 + 1/(k₀h · a)) / h

    Maximum relative error versus the exact dispersion relation is < 0.2 %
    across all depth regimes (deep, intermediate, and shallow water), which
    is negligible compared with typical uncertainties in Hs and Tp
    (≥ 5–10 %).  Cells where h ≤ 0 propagate NaN.

    Args:
        omega: Angular frequency DataArray (rad/s).
        h: Water depth DataArray (m, positive downward).  Broadcast-
            compatible with omega.
        g: Gravitational acceleration (m/s², default 9.81).

    Returns:
        Wavenumber DataArray k (rad/m).  NaN where h ≤ 0.
    """
    h_safe = h.where(h > 0)
    k0h = omega ** 2 * h_safe / g
    a = (1.0
         + 0.6522 * k0h
         + 0.4622 * k0h ** 2
         + 0.0864 * k0h ** 4
         + 0.0675 * k0h ** 5)
    return k0h * (1.0 + 1.0 / (k0h * a)) ** 0.5 / h_safe


def _wavenumber_nr(
    omega: xr.DataArray,
    h: xr.DataArray,
    g: float = 9.81,
    n_iter: int = 50,
) -> xr.DataArray:
    """Exact wavenumber via Newton-Raphson iteration (dask-safe via apply_ufunc).

    Solves ω² = g k tanh(kh) iteratively to full float64 precision.  The
    Python ``for`` loop is confined inside a numpy function dispatched by
    ``apply_ufunc``, so the dask graph remains flat (one task per chunk
    regardless of iteration count).

    Args:
        omega: Angular frequency DataArray (rad/s).
        h: Water depth DataArray (m, positive downward).
        g: Gravitational acceleration (m/s²).
        n_iter: Newton-Raphson iterations (default 50; typically converges in
            fewer than 20 for any physically realistic depth).

    Returns:
        Wavenumber DataArray k (rad/m).  NaN where h ≤ 0.
    """
    def _np_solve(om: np.ndarray, h_: np.ndarray) -> np.ndarray:
        valid = h_ > 0
        h_safe = np.where(valid, h_, 1.0)
        k = om ** 2 / g
        for _ in range(n_iter):
            kh = k * h_safe
            tanh_kh = np.tanh(kh)
            f  = om ** 2 - g * k * tanh_kh
            fp = -g * (tanh_kh + kh / np.cosh(kh) ** 2)
            k -= f / fp
        return np.where(valid, k, np.nan)

    return xr.apply_ufunc(
        _np_solve,
        omega,
        h,
        dask="parallelized",
        output_dtypes=["float64"],
    )


@register_derived("uorb")
def uorb(
    ds: xr.Dataset,
    *,
    hs: str = "hs",
    tp: str = "tp",
    depth: str | float = "depth",
    z: str | float = 0.0,
    reference: str = "bed",
    g: float = 9.81,
    solver: str = "explicit",
) -> xr.DataArray:
    """Significant horizontal wave orbital velocity at position *z*.

    Monochromatic approximation (Soulsby 1997) using integrated spectral
    parameters Hs and a representative period T:

    .. math::

        U(z) = \\frac{\\pi H_s}{T} \\cdot \\frac{\\cosh(k z_{bed})}{\\sinh(k h)}

    where *k* solves the linear dispersion relation ω² = g k tanh(kh) and
    z_bed is the height above the seabed.  See the derived variables
    documentation for period choice, depth profiles, solver comparison,
    and accuracy notes.

    Args:
        ds: Input dataset.
        hs: Name of the significant wave height variable (m).
        tp: Name of the representative wave period variable (s).
        depth: Name of a depth variable (m, positive downward) or a scalar
            depth to apply uniformly across the domain.
        z: Position at which to evaluate orbital velocity (m, default 0).
            Variable name or scalar float.  Interpretation depends on
            ``reference``: height above the seabed when ``reference='bed'``,
            or depth below the still-water surface when ``reference='surface'``.
        reference: Reference datum for *z*:

            * ``'bed'`` *(default)* — *z* is height above the seabed
              (0 = seabed, h = surface).
            * ``'surface'`` — *z* is depth below the still-water surface
              (0 = surface, h = seabed).  Converted internally via
              z_bed = depth − z, so spatially varying bathymetry is handled
              automatically.

        g: Gravitational acceleration (m/s², default 9.81).
        solver: ``'explicit'`` (default, Chen & Thomson 1993, < 0.2 % error,
            ~40× faster) or ``'exact'`` (Newton-Raphson, float64 precision).

    Returns:
        Horizontal orbital velocity DataArray (m/s, float32).  NaN where
        depth ≤ 0 or input variables contain NaN.

    Raises:
        ValueError: If *solver* is not ``'explicit'`` or ``'exact'``.
        ValueError: If *reference* is not ``'bed'`` or ``'surface'``.

    References:
        Soulsby, R. L. (1997). *Dynamics of Marine Sands*. Thomas Telford.
        Chen, G., & Thomson, J. (1993). *Ocean Engineering*, 20(6), 487–507.
    """
    if solver not in ("explicit", "exact"):
        raise ValueError(f"solver must be 'explicit' or 'exact', got {solver!r}")
    if reference not in ("bed", "surface"):
        raise ValueError(f"reference must be 'bed' or 'surface', got {reference!r}")

    hs_da = ds[hs].astype("float64")
    tp_da = ds[tp].astype("float64")

    if isinstance(depth, str):
        h_da = ds[depth].astype("float64")
    else:
        h_da = xr.full_like(tp_da, fill_value=float(depth), dtype="float64")

    z_val: xr.DataArray | float
    if isinstance(z, str):
        z_val = ds[z].astype("float64")
    else:
        z_val = float(z)

    if reference == "surface":
        z_val = h_da - z_val

    omega = 2.0 * np.pi / tp_da

    if solver == "explicit":
        k = _wavenumber(omega, h_da, g=g)
    else:
        k = _wavenumber_nr(omega, h_da, g=g)

    kh = k * h_da
    kz = k * z_val
    transfer = (np.exp(kz - kh) + np.exp(-kz - kh)) / (1.0 - np.exp(-2.0 * kh))

    out = (np.pi * hs_da / tp_da * transfer).astype("float32")

    if reference == "surface":
        z_label = (
            f"z = {z} m below surface" if isinstance(z, (int, float))
            else f"z = '{z}' m below surface"
        )
    else:
        z_label = (
            f"z = {z} m above seabed" if isinstance(z, (int, float))
            else f"z = '{z}' m above seabed"
        )
    out.attrs = {
        "standard_name": "sea_surface_wave_orbital_velocity",
        "long_name": f"significant horizontal wave orbital velocity at {z_label}",
        "units": "m/s",
    }
    return out
