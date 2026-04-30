"""Derived wind variables."""
from __future__ import annotations

import numpy as np
import xarray as xr

from gridstats.registry import register_derived


@register_derived("wspd")
def wspd(
    ds: xr.Dataset,
    *,
    uwnd: str = "uwnd",
    vwnd: str = "vwnd",
) -> xr.DataArray:
    """Wind speed from eastward/northward components.

    Args:
        ds: Input dataset.
        uwnd: Name of the eastward wind component variable.
        vwnd: Name of the northward wind component variable.

    Returns:
        Wind speed DataArray (m/s).
    """
    out = np.sqrt(ds[uwnd] ** 2 + ds[vwnd] ** 2)
    out.attrs = {
        "standard_name": "wind_speed",
        "long_name": "wind speed",
        "units": "m/s",
    }
    return out


@register_derived("wdir")
def wdir(
    ds: xr.Dataset,
    *,
    uwnd: str = "uwnd",
    vwnd: str = "vwnd",
) -> xr.DataArray:
    """Wind coming-from direction from eastward/northward components.

    Args:
        ds: Input dataset.
        uwnd: Name of the eastward wind component variable.
        vwnd: Name of the northward wind component variable.

    Returns:
        Wind direction DataArray (degrees, coming-from meteorological convention).
    """
    out = (270.0 - np.degrees(np.arctan2(ds[vwnd], ds[uwnd]))) % 360.0
    out.attrs = {
        "standard_name": "wind_from_direction",
        "long_name": "wind from direction",
        "units": "degree",
    }
    return out
