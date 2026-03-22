"""Directional sector statistics."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr

from onstats.registry import register_stat

logger = logging.getLogger(__name__)


@register_stat("statdir")
def statdir(
    data: xr.Dataset,
    *,
    funcs: list[str],
    dir_var: str = "dpm",
    nsector: int = 4,
    dim: str = "time",
    **kwargs: Any,
) -> xr.Dataset:
    """Apply multiple stat functions over directional sectors.

    Bins the dataset by ``nsector`` equal directional sectors, applies each
    function in ``funcs`` to each sector, then concatenates results along a
    new 'direction' dimension.

    Args:
        data: Input dataset. Must contain ``dir_var``.
        funcs: Names of registered stat functions to apply per sector.
        dir_var: Name of the directional variable used for binning.
        nsector: Number of equally-spaced directional sectors (default 4).
        dim: Time dimension name.
        **kwargs: Forwarded to each stat function.

    Returns:
        Dataset with a 'direction' dimension containing sector-centre values.
    """
    from onstats.registry import get_stat

    if dir_var not in data:
        raise ValueError(
            f"Directional variable '{dir_var}' not found in dataset. "
            f"Available: {list(data.data_vars)}"
        )

    dirs = data[dir_var]
    dsector = 360.0 / nsector
    sector_centres = np.linspace(0, 360 - dsector, nsector)
    starts = (sector_centres - dsector / 2) % 360
    stops = (sector_centres + dsector / 2) % 360

    sector_results = []
    for start, stop in zip(starts, stops):
        logger.info("statdir: sector [%.1f, %.1f)", start, stop)
        if stop > start:
            mask = (dirs >= start) & (dirs < stop)
        else:
            mask = (dirs >= start) | (dirs < stop)
        sector_ds = data.where(mask)

        func_results = []
        for func_name in funcs:
            fn = get_stat(func_name)
            func_results.append(fn(sector_ds, dim=dim, **kwargs))
        sector_results.append(xr.merge(func_results))

    dsout = xr.concat(sector_results, dim="direction").assign_coords(
        {"direction": sector_centres}
    )
    dsout["direction"].attrs = {
        "standard_name": dirs.attrs.get("standard_name", "direction") + "_sector",
        "long_name": dirs.attrs.get("long_name", "direction") + " sector",
        "units": dirs.attrs.get("units", "degree"),
    }
    return dsout
