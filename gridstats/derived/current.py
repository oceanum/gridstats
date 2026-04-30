"""Derived ocean current variables."""
from __future__ import annotations

import numpy as np
import xarray as xr

from gridstats.registry import register_derived


@register_derived("cspd")
def cspd(
    ds: xr.Dataset,
    *,
    ucur: str = "ucur",
    vcur: str = "vcur",
) -> xr.DataArray:
    """Sea water speed from eastward/northward current components.

    Args:
        ds: Input dataset.
        ucur: Name of the eastward current component variable.
        vcur: Name of the northward current component variable.

    Returns:
        Current speed DataArray (m/s).
    """
    out = np.sqrt(ds[ucur] ** 2 + ds[vcur] ** 2)
    out.attrs = {
        "standard_name": "sea_water_speed",
        "long_name": "sea water speed",
        "units": "m/s",
    }
    return out


@register_derived("cdir")
def cdir(
    ds: xr.Dataset,
    *,
    ucur: str = "ucur",
    vcur: str = "vcur",
) -> xr.DataArray:
    """Sea water going-to direction from eastward/northward current components.

    Args:
        ds: Input dataset.
        ucur: Name of the eastward current component variable.
        vcur: Name of the northward current component variable.

    Returns:
        Current direction DataArray (degrees, going-to oceanographic convention).
    """
    out = (90.0 - np.degrees(np.arctan2(ds[vcur], ds[ucur]))) % 360.0
    out.attrs = {
        "standard_name": "direction_of_sea_water_velocity",
        "long_name": "direction of sea water velocity",
        "units": "degree",
    }
    return out
