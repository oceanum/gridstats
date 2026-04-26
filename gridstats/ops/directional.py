"""Directional sector statistics."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr

from gridstats.registry import register_stat

logger = logging.getLogger(__name__)


def _modal_direction_da(
    direction: xr.DataArray,
    weight: xr.DataArray | None,
    dim: str,
    bin_width_deg: float,
    smooth: bool,
) -> xr.DataArray:
    """Compute modal direction for a single DataArray over *dim*."""
    if 360.0 % bin_width_deg != 0:
        raise ValueError("bin_width_deg must divide 360 evenly.")

    n_bins = int(round(360.0 / bin_width_deg))
    bin_edges = np.linspace(0.0, 360.0, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    dir_wrapped = direction % 360.0
    bin_idx = (dir_wrapped // bin_width_deg).astype("int16")
    bin_idx = bin_idx.where(bin_idx < n_bins, 0)

    if weight is None:
        w = xr.ones_like(direction, dtype="float32")
    else:
        w = weight.astype("float32")

    valid = direction.notnull() & w.notnull()
    w = w.where(valid, 0.0)

    hist_layers = []
    for b in range(n_bins):
        hist_layers.append(w.where(bin_idx == b, 0.0).sum(dim=dim))
    hist = xr.concat(hist_layers, dim="bin").assign_coords(bin=bin_centres)

    if smooth:
        padded = xr.concat(
            [hist.isel(bin=[-1]), hist, hist.isel(bin=[0])], dim="bin"
        )
        smoothed = padded.rolling(bin=3, center=True).mean().isel(bin=slice(1, -1))
        hist = smoothed.assign_coords(bin=bin_centres)

    winning = hist.argmax(dim="bin")
    modal = xr.DataArray(bin_centres, dims="bin").isel(bin=winning)

    total = hist.sum(dim="bin")
    return modal.where(total > 0)


@register_stat("modal_direction")
def modal_direction(
    data: xr.Dataset,
    *,
    dim: str = "time",
    group: str | None = None,
    weight_var: str | None = None,
    bin_width_deg: float = 10.0,
    smooth: bool = True,
    **kwargs: Any,
) -> xr.Dataset:
    """Per-cell modal (most frequent) direction from a weighted circular histogram.

    Uses a histogram-based mode rather than a vector mean, so it is robust to
    bimodal and anti-parallel direction distributions (e.g. monsoon reversals
    where the arithmetic mean of 90° and 270° is meaningless).

    Args:
        data: Input dataset. Direction variables must be in degrees in [0, 360).
        dim: Dimension to reduce along (default ``'time'``).
        group: Time component for grouped climatology: ``'month'``, ``'season'``,
            or ``'year'``. When set, the output gains a dimension named after the
            group (e.g. a ``'month'`` dimension with values 1–12).
        weight_var: Name of a variable in *data* to use as histogram weights.
            Typical choices: ``'hs'`` for occurrence, ``'hs'**2`` for energy,
            ``'hs'**2 * 'te'`` for energy flux. When ``None`` each sample
            contributes equally (unweighted frequency histogram).
        bin_width_deg: Histogram bin width in degrees. Must divide 360 evenly.
            10° (36 bins) is a good default; use 5° for sharper modes on long
            records.
        smooth: If ``True`` (default), apply a 3-bin circular moving average
            to the histogram before taking the argmax, stabilising the mode
            against single-bin noise.
        **kwargs: Accepted but not forwarded (pipeline compatibility).

    Returns:
        Dataset with one variable per direction variable in *data* (excluding
        *weight_var*). Each variable holds the centre of the dominant histogram
        bin at every grid cell. Gains a group dimension when *group* is set.

    Note:
        For large grids set ``tiles:`` on the call to bound peak memory — each
        bin pass requires a full-grid ``where`` + ``sum``.
    """
    weight = data[weight_var] if weight_var is not None else None
    dir_vars = [v for v in data.data_vars if v != weight_var]

    if group is not None:
        grouped = data.groupby(f"time.{group}")
        group_results = []
        group_keys = []
        for key, group_ds in grouped:
            w = group_ds[weight_var] if weight_var is not None else None
            out_vars = {
                v: _modal_direction_da(group_ds[v], w, dim, bin_width_deg, smooth)
                for v in dir_vars
            }
            group_results.append(xr.Dataset(out_vars))
            group_keys.append(key)
        dsout = xr.concat(group_results, dim=group)
        dsout[group] = group_keys
        return dsout

    out_vars = {
        v: _modal_direction_da(data[v], weight, dim, bin_width_deg, smooth)
        for v in dir_vars
    }
    return xr.Dataset(out_vars)


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
    from gridstats.registry import get_stat

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
