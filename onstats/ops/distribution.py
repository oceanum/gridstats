"""Joint distribution (histogram) operations."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from onstats.registry import register_stat

logger = logging.getLogger(__name__)

_FILLVALUE = int(-(2**31))


def _make_bins(spec: dict[str, Any] | list | np.ndarray, darr: xr.DataArray) -> np.ndarray:
    """Build a bin-edge array from a dict spec or a plain array.

    Dict keys: start, stop (optional), step, dtype (optional).
    If stop is omitted it is inferred from the data maximum.
    """
    if isinstance(spec, (list, np.ndarray)):
        return np.asarray(spec)
    stop = spec.get("stop", float(darr.max().values) + spec.get("step", 1))
    return np.arange(spec["start"], stop + spec.get("step", 1), spec["step"])


def _wrap_directions(darr: xr.DataArray, dirmax: float) -> xr.DataArray:
    """Wrap directions > dirmax to negative values for correct histogram binning."""
    return darr.where(darr <= dirmax, darr - 360)


def _np_histogram2d(arr1: np.ndarray, arr2: np.ndarray, bins1: np.ndarray, bins2: np.ndarray) -> np.ndarray:
    hist, _, _ = np.histogram2d(arr1, arr2, bins=[bins1, bins2])
    return hist.astype("float32")


def _np_histogram3d(
    arr1: np.ndarray, arr2: np.ndarray, arr3: np.ndarray,
    bins1: np.ndarray, bins2: np.ndarray, bins3: np.ndarray,
) -> np.ndarray:
    hist, _ = np.histogramdd([arr1, arr2, arr3], bins=[bins1, bins2, bins3])
    return hist.astype("float32")


def _compute_distribution3(
    data: xr.Dataset,
    var1: str, var2: str, var3: str,
    bins1: np.ndarray, bins2: np.ndarray, bins3: np.ndarray,
    isdir3: bool,
    dim: str,
) -> xr.DataArray:
    """Compute a 3-D joint histogram over a single dataset."""
    d1 = data[var1]
    d2 = data[var2]
    d3 = _wrap_directions(data[var3], bins3.max()) if isdir3 else data[var3]

    return xr.apply_ufunc(
        _np_histogram3d,
        d1, d2, d3,
        kwargs={"bins1": bins1, "bins2": bins2, "bins3": bins3},
        input_core_dims=[[dim], [dim], [dim]],
        output_core_dims=[[var1, var2, var3]],
        exclude_dims={dim},
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"],
        dask_gufunc_kwargs={
            "output_sizes": {
                var1: len(bins1) - 1,
                var2: len(bins2) - 1,
                var3: len(bins3) - 1,
            },
            "allow_rechunk": True,
        },
    )


@register_stat("distribution3")
def distribution3(
    data: xr.Dataset,
    *,
    dim: str = "time",
    var1: str = "hs",
    var2: str = "tp",
    var3: str = "dpm",
    bins1: dict | list = {"start": 0, "step": 0.5},
    bins2: dict | list = {"start": 0, "step": 1.0},
    bins3: dict | list = {"start": 0, "stop": 360, "step": 45},
    isdir1: bool = False,
    isdir2: bool = False,
    isdir3: bool = True,
    group: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Compute a 3-D joint histogram (e.g. Hs × Tp × Dir).

    Args:
        data: Input dataset containing var1, var2, var3.
        dim: Time dimension name.
        var1: Name of the first variable (e.g. 'hs').
        var2: Name of the second variable (e.g. 'tp').
        var3: Name of the third variable, often directional (e.g. 'dpm').
        bins1/2/3: Bin specifications as dicts with 'start', 'stop', 'step'
            or as plain arrays.
        isdir1/2/3: Whether each variable is directional (wraps values > max
            to negative to avoid discontinuity at 0/360).
        group: Time component to group by (e.g. 'month').

    Returns:
        Dataset with variable 'dist' and dimensions (var1, var2, var3) plus
        bin-centre coordinates.
    """
    b1 = _make_bins(bins1, data[var1])
    b2 = _make_bins(bins2, data[var2])
    b3 = _make_bins(bins3, data[var3])

    if group is not None:
        groups = data.groupby(f"{dim}.{group}")
        parts = []
        for key, ds in groups:
            hist = _compute_distribution3(ds, var1, var2, var3, b1, b2, b3, isdir3, dim)
            parts.append(hist.expand_dims({group: [key]}))
        hist = xr.concat(parts, dim=group)
    else:
        hist = _compute_distribution3(data, var1, var2, var3, b1, b2, b3, isdir3, dim)

    centres1 = 0.5 * (b1[:-1] + b1[1:])
    centres2 = 0.5 * (b2[:-1] + b2[1:])
    centres3 = 0.5 * (b3[:-1] + b3[1:])
    hist = hist.assign_coords({var1: centres1, var2: centres2, var3: centres3})
    hist.encoding["_FillValue"] = _FILLVALUE
    return hist.rename("dist").to_dataset()


@register_stat("distribution2")
def distribution2(
    data: xr.Dataset,
    *,
    dim: str = "time",
    var1: str = "wspd",
    var2: str = "wdir",
    bins1: dict | list = {"start": 0, "step": 1.0},
    bins2: dict | list = {"start": 0, "stop": 360, "step": 45},
    isdir1: bool = False,
    isdir2: bool = True,
    group: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Compute a 2-D joint histogram (e.g. speed × direction).

    Args:
        data: Input dataset containing var1, var2.
        dim: Time dimension name.
        var1: Name of the first variable (e.g. 'wspd').
        var2: Name of the second variable, often directional (e.g. 'wdir').
        bins1/2: Bin specifications.
        isdir1/2: Whether each variable is directional.
        group: Time component to group by (e.g. 'month').

    Returns:
        Dataset with variable 'dist2' and dimensions (var1, var2).
    """
    b1 = _make_bins(bins1, data[var1])
    b2 = _make_bins(bins2, data[var2])
    d1 = _wrap_directions(data[var1], b1.max()) if isdir1 else data[var1]
    d2 = _wrap_directions(data[var2], b2.max()) if isdir2 else data[var2]

    def _compute(ds1, ds2):
        return xr.apply_ufunc(
            _np_histogram2d,
            ds1, ds2,
            kwargs={"bins1": b1, "bins2": b2},
            input_core_dims=[[dim], [dim]],
            output_core_dims=[[var1, var2]],
            exclude_dims={dim},
            vectorize=True,
            dask="parallelized",
            output_dtypes=["float32"],
            dask_gufunc_kwargs={
                "output_sizes": {var1: len(b1) - 1, var2: len(b2) - 1},
                "allow_rechunk": True,
            },
        )

    if group is not None:
        g1, g2 = d1.groupby(f"{dim}.{group}"), d2.groupby(f"{dim}.{group}")
        parts = [
            _compute(v1, v2).expand_dims({group: [k]})
            for (k, v1), (_, v2) in zip(g1, g2)
        ]
        hist = xr.concat(parts, dim=group)
    else:
        hist = _compute(d1, d2)

    centres1 = 0.5 * (b1[:-1] + b1[1:])
    centres2 = 0.5 * (b2[:-1] + b2[1:])
    hist = hist.assign_coords({var1: centres1, var2: centres2})
    hist.encoding["_FillValue"] = _FILLVALUE
    return hist.rename("dist2").to_dataset()


@register_stat("distribution3_timestep")
def distribution3_timestep(
    data: xr.Dataset,
    *,
    dim: str = "time",
    var1: str = "hs",
    var2: str = "tp",
    var3: str = "dpm",
    bins1: dict | list = {"start": 0, "step": 0.5},
    bins2: dict | list = {"start": 0, "step": 1.0},
    bins3: dict | list = {"start": 0, "stop": 360, "step": 45},
    isdir1: bool = False,
    isdir2: bool = False,
    isdir3: bool = True,
    freq: str = "30d",
    group: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """3-D joint histogram computed in time chunks to limit memory use.

    Accumulates histogram counts over successive time windows of size `freq`
    then sums them. Useful for very long timeseries that cannot be fully
    rechunked into memory.

    Args:
        data: Input dataset containing var1, var2, var3.
        dim: Time dimension name.
        var1/2/3: Variable names.
        bins1/2/3: Bin specifications.
        isdir1/2/3: Whether each variable is directional.
        freq: Pandas-compatible frequency string for time chunking (e.g. '30d').
        group: Time component to group by (e.g. 'month').

    Returns:
        Dataset with accumulated joint distribution counts.
    """
    b1 = _make_bins(bins1, data[var1])
    b2 = _make_bins(bins2, data[var2])
    b3 = _make_bins(bins3, data[var3])

    times = pd.DatetimeIndex(data[dim].values)
    windows = pd.date_range(start=times[0], end=times[-1], freq=freq)
    if len(windows) == 0 or windows[-1] < times[-1]:
        windows = windows.append(pd.DatetimeIndex([times[-1]]))

    dsout = None
    total_windows = len(windows) - 1
    for i, (t0, t1) in enumerate(zip(windows[:-1], windows[1:])):
        logger.info("distribution3_timestep: window %d/%d (%s – %s)", i + 1, total_windows, t0, t1)
        chunk = data.sel({dim: slice(t0, t1)})
        chunk = chunk.load()
        hist = _compute_distribution3(chunk, var1, var2, var3, b1, b2, b3, isdir3, dim)
        hist = hist.compute()
        dsout = hist if dsout is None else dsout + hist

    if dsout is None:
        dsout = _compute_distribution3(data, var1, var2, var3, b1, b2, b3, isdir3, dim)
        dsout = dsout.compute()

    centres1 = 0.5 * (b1[:-1] + b1[1:])
    centres2 = 0.5 * (b2[:-1] + b2[1:])
    centres3 = 0.5 * (b3[:-1] + b3[1:])
    dsout = dsout.assign_coords({var1: centres1, var2: centres2, var3: centres3})
    dsout.encoding["_FillValue"] = _FILLVALUE
    return dsout.rename("dist").to_dataset()
