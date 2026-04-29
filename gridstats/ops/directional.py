"""Directional sector statistics."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr

from gridstats.registry import register_stat

logger = logging.getLogger(__name__)


def _np_modal_direction(
    direction: np.ndarray,
    weight: np.ndarray,
    n_bins: int,
    bin_width_deg: float,
    bin_centres: np.ndarray,
    smooth: bool,
) -> np.float32:
    """Compute modal direction from 1-D direction and weight arrays.

    Returns NaN when all values are missing or total weight is zero.
    """
    valid = np.isfinite(direction) & np.isfinite(weight)
    d = direction[valid] % 360.0
    w = weight[valid]

    if d.size == 0:
        return np.float32(np.nan)

    bin_idx = (d // bin_width_deg).astype(int)
    np.clip(bin_idx, 0, n_bins - 1, out=bin_idx)

    hist = np.zeros(n_bins, dtype="float32")
    np.add.at(hist, bin_idx, w)

    if hist.sum() == 0:
        return np.float32(np.nan)

    if smooth:
        padded = np.concatenate([hist[-1:], hist, hist[:1]])
        hist = np.convolve(padded, np.ones(3, dtype="float32") / 3, mode="valid")

    return bin_centres[np.argmax(hist)].astype("float32")


def _apply_modal_direction(
    direction: xr.DataArray,
    weight: xr.DataArray,
    dim: str,
    n_bins: int,
    bin_width_deg: float,
    bin_centres: np.ndarray,
    smooth: bool,
) -> xr.DataArray:
    """Apply modal direction computation via apply_ufunc (dask-compatible)."""
    return xr.apply_ufunc(
        _np_modal_direction,
        direction,
        weight,
        kwargs={
            "n_bins": n_bins,
            "bin_width_deg": bin_width_deg,
            "bin_centres": bin_centres,
            "smooth": smooth,
        },
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[]],
        exclude_dims={dim},
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


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
        ``apply_ufunc`` with ``allow_rechunk=True`` handles the single-chunk
        requirement along *dim* automatically in the dask graph, so manual
        rechunking is not needed before calling this stat.
    """
    if 360.0 % bin_width_deg != 0:
        raise ValueError("bin_width_deg must divide 360 evenly.")

    n_bins = int(round(360.0 / bin_width_deg))
    bin_edges = np.linspace(0.0, 360.0, n_bins + 1)
    bin_centres = (0.5 * (bin_edges[:-1] + bin_edges[1:])).astype("float32")

    dir_vars = [v for v in data.data_vars if v != weight_var]

    def _compute(ds: xr.Dataset) -> xr.Dataset:
        w = ds[weight_var].astype("float32") if weight_var is not None else xr.ones_like(ds[dir_vars[0]], dtype="float32")
        return xr.Dataset(
            {
                v: _apply_modal_direction(
                    ds[v], w, dim, n_bins, bin_width_deg, bin_centres, smooth
                )
                for v in dir_vars
            }
        )

    if group is not None:
        parts = [
            _compute(group_ds).expand_dims({group: [key]})
            for key, group_ds in data.groupby(f"{dim}.{group}")
        ]
        return xr.concat(parts, dim=group)

    return _compute(data)


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
