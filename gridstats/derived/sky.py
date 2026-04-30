"""Derived sky condition variables."""
from __future__ import annotations

import xarray as xr

from gridstats.registry import register_derived


@register_derived("clear_sky")
def clear_sky(
    ds: xr.Dataset,
    *,
    cloud_cover: str = "cloud_cover",
    cover_threshold: float = 0.0,
) -> xr.DataArray:
    """Boolean mask for clear-sky conditions.

    Args:
        ds: Input dataset.
        cloud_cover: Name of the cloud area fraction variable (0–1).
        cover_threshold: Maximum cloud fraction considered clear sky.

    Returns:
        Boolean DataArray: True where sky is clear.
    """
    out = ds[cloud_cover] <= cover_threshold
    out.attrs = {"standard_name": "clear_sky", "long_name": "clear sky", "units": ""}
    return out


@register_derived("covered_sky")
def covered_sky(
    ds: xr.Dataset,
    *,
    cloud_cover: str = "cloud_cover",
    cover_threshold: float = 1.0,
) -> xr.DataArray:
    """Boolean mask for fully overcast conditions.

    Args:
        ds: Input dataset.
        cloud_cover: Name of the cloud area fraction variable (0–1).
        cover_threshold: Minimum cloud fraction considered covered sky.

    Returns:
        Boolean DataArray: True where sky is fully covered.
    """
    out = ds[cloud_cover] >= cover_threshold
    out.attrs = {
        "standard_name": "covered_sky",
        "long_name": "covered sky",
        "units": "",
    }
    return out
