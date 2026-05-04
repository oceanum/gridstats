"""Derived wave parameter variables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from gridstats.registry import register_derived

DOUGLAS_SEA_BOUNDS = [0.0, 0.1, 0.5, 1.25, 2.5, 4.0, 6.0, 9.0, 14.0, np.inf]
# np.digitize edges: same bounds without the trailing inf (digitize handles >14 as index 9).
_SEA_EDGES = np.array(DOUGLAS_SEA_BOUNDS[:-1])

# Degree 9 ("confused") is a crossing-seas condition, not a height/wavelength
# threshold, so it is not listed here. Apply it via the crossing_seas derived
# variable when a secondary swell partition is available.
DOUGLAS_SWELL_INTERVALS = {
    0: {"height": pd.Interval(-np.inf, 0.0), "length": pd.Interval(-np.inf, np.inf)},
    1: {"height": pd.Interval(0.0, 2.0),     "length": pd.Interval(0.0, 200.0)},
    2: {"height": pd.Interval(0.0, 2.0),     "length": pd.Interval(200.0, np.inf)},
    3: {"height": pd.Interval(2.0, 4.0),     "length": pd.Interval(0.0, 100.0)},
    4: {"height": pd.Interval(2.0, 4.0),     "length": pd.Interval(100.0, 200.0)},
    5: {"height": pd.Interval(2.0, 4.0),     "length": pd.Interval(200.0, np.inf)},
    6: {"height": pd.Interval(4.0, np.inf),  "length": pd.Interval(0.0, 100.0)},
    7: {"height": pd.Interval(4.0, np.inf),  "length": pd.Interval(100.0, 200.0)},
    8: {"height": pd.Interval(4.0, np.inf),  "length": pd.Interval(200.0, np.inf)},
}

# Derived from DOUGLAS_SWELL_INTERVALS for the apply_ufunc path.
# Rows = hs band (low 0–2 m / moderate 2–4 m / high >4 m).
# Cols = lp band (short ≤100 m / average 100–200 m / long >200 m).
_HS_SWELL_EDGES = np.array([0.0, 2.0, 4.0])   # right=True → bands 0–3
_LP_SWELL_EDGES = np.array([100.0, 200.0])     # right=True → bands 0–2
_SWELL_LUT = np.array([
    [1, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
], dtype=np.float32)


def _classify_swell(hs_arr: np.ndarray, lp_arr: np.ndarray) -> np.ndarray:
    h = np.digitize(hs_arr, _HS_SWELL_EDGES, right=True)  # 0=no-swell, 1–3=height band
    l = np.digitize(lp_arr, _LP_SWELL_EDGES, right=True)  # 0–2=length band
    out = _SWELL_LUT[np.clip(h - 1, 0, 2), l]             # fancy index → copy
    out[h == 0] = 0.0                                       # hs ≤ 0: no swell
    out[lp_arr <= 0.0] = 0.0                               # lp ≤ 0: unclassifiable
    return out


@register_derived("tp")
def tp(
    ds: xr.Dataset,
    *,
    fp: str = "fp",
) -> xr.DataArray:
    """Peak wave period from peak wave frequency.

    Args:
        ds: Input dataset.
        fp: Name of the peak wave frequency variable (Hz).

    Returns:
        Peak wave period DataArray (s).
    """
    out = 1.0 / ds[fp]
    out.attrs = {
        "standard_name": "sea_surface_wave_period_at_variance_spectral_density_maximum",
        "long_name": "peak wave period of sea and swell waves",
        "units": "s",
    }
    return out


@register_derived("douglas_sea")
def douglas_sea(
    ds: xr.Dataset,
    *,
    hs_sea: str = "hs_sea",
) -> xr.DataArray:
    """Douglas sea scale (0–9) from wind-sea significant wave height.

    Scale 0 = glassy; scale 9 = phenomenal (Hs > 14 m).

    Args:
        ds: Input dataset.
        hs_sea: Name of the wind-sea significant wave height variable (m).

    Returns:
        Douglas sea scale DataArray (integer-valued float32, 0–9).
    """
    arr = ds[hs_sea]
    out = xr.apply_ufunc(
        np.digitize,
        arr,
        kwargs={"bins": _SEA_EDGES, "right": True},
        dask="parallelized",
        output_dtypes=[np.intp],
    ).astype("float32")
    # np.digitize returns len(bins)=9 for NaN inputs; restore to 0 to match
    # the fill_value=0 behaviour of the previous loop-based implementation.
    out = out.where(arr.notnull(), other=0.0)
    out.attrs = {
        "standard_name": "sea_surface_wave_douglas_sea_scale",
        "long_name": "douglas sea scale",
        "units": "",
    }
    return out


@register_derived("douglas_swell")
def douglas_swell(
    ds: xr.Dataset,
    *,
    hs_sw1: str = "hs_sw1",
    lp_sw1: str = "lp_sw1",
) -> xr.DataArray:
    """Douglas swell scale (0–9) from primary swell height and wavelength.

    Args:
        ds: Input dataset.
        hs_sw1: Name of the primary swell significant wave height variable (m).
        lp_sw1: Name of the primary swell peak wavelength variable (m).

    Returns:
        Douglas swell scale DataArray (integer-valued float32, 0–9).
    """
    hs = ds[hs_sw1]
    lp = ds[lp_sw1]
    out = xr.apply_ufunc(
        _classify_swell,
        hs,
        lp,
        dask="parallelized",
        output_dtypes=[np.float32],
    )
    out = out.where(hs.notnull() & lp.notnull(), other=0.0)
    out.attrs = {
        "standard_name": "sea_surface_wave_douglas_swell_scale",
        "long_name": "douglas swell scale",
        "units": "",
    }
    return out


@register_derived("crossing_seas")
def crossing_seas(
    ds: xr.Dataset,
    *,
    hs: str = "hs",
    hs_sea: str = "hs_sea",
    hs_sw1: str = "hs_sw1",
    dir_sea: str = "dir_sea",
    dir_sw1: str = "dir_sw1",
    hs_sw2: str | None = None,
    dir_sw2: str | None = None,
    hs_threshold: float = 0.0,
    angle_threshold: float = 40.0,
    energy_fraction: float = 0.2,
) -> xr.DataArray:
    """Boolean mask indicating crossing-seas conditions.

    Crossing seas are identified when:
    (1) The relative angle between two wave systems exceeds ``angle_threshold``.
    (2) The less energetic system carries at least ``energy_fraction`` of total energy
        (i.e. Hs_minor > sqrt(energy_fraction) * Hs_total).

    Args:
        ds: Input dataset.
        hs: Total significant wave height variable name (m).
        hs_sea: Wind-sea significant wave height variable name (m).
        hs_sw1: Primary swell significant wave height variable name (m).
        dir_sea: Wind-sea direction variable name (degrees).
        dir_sw1: Primary swell direction variable name (degrees).
        hs_sw2: Secondary swell Hs variable name. Set to enable sea/sw2 and sw1/sw2
            pair checks.
        dir_sw2: Secondary swell direction variable name.
        hs_threshold: Minimum total Hs (m) below which crossing seas are not reported.
        angle_threshold: Minimum relative angle (degrees) between two systems.
        energy_fraction: Minimum energy fraction (relative to total Hs) for the
            weaker system.

    Returns:
        Boolean DataArray: True where crossing seas are detected.

    Reference:
        Li, X.M. (2016). A new insight from space into swell propagation and
        crossing in the global oceans. Geophysical Research Letters, 43(10).
    """
    def _angle(d1: xr.DataArray, d2: xr.DataArray) -> xr.DataArray:
        diff = np.abs(d1 % 360 - d2 % 360)
        return np.minimum(diff, 360 - diff)

    hs_tot = ds[hs].where(ds[hs] > hs_threshold)
    min_hs = np.sqrt(energy_fraction) * hs_tot

    hs_s = ds[hs_sea]
    hs_w1 = ds[hs_sw1]
    d_s = ds[dir_sea]
    d_w1 = ds[dir_sw1]

    c1 = (hs_s > min_hs) & (hs_w1 > min_hs) & (_angle(d_s, d_w1) > angle_threshold)
    result = c1

    if hs_sw2 is not None and dir_sw2 is not None:
        hs_w2 = ds[hs_sw2]
        d_w2 = ds[dir_sw2]
        c2 = (hs_s > min_hs) & (hs_w2 > min_hs) & (_angle(d_s, d_w2) > angle_threshold)
        c3 = (hs_w1 > min_hs) & (hs_w2 > min_hs) & (_angle(d_w1, d_w2) > angle_threshold)
        result = c1 | c2 | c3

    result.attrs = {
        "standard_name": "sea_surface_crossing_seas",
        "long_name": "crossing seas",
        "units": "",
    }
    return result
