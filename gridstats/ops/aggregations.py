"""Standard xarray aggregation operations."""
from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

from gridstats.registry import register_stat


def _groupby(
    data: xr.Dataset, group: str | None
) -> xr.Dataset | xr.core.groupby.DatasetGroupBy:
    """Apply groupby on the time dimension if group is provided."""
    if group is not None:
        return data.groupby(f"time.{group}")
    return data


@register_stat("mean")
def mean(data: xr.Dataset, *, dim: str = "time", group: str | None = None, **kwargs: Any) -> xr.Dataset:
    """Arithmetic mean along a dimension.

    Wraps [`xr.Dataset.mean`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.mean.html).
    Any extra keyword arguments are forwarded to xarray.

    Args:
        data: Input dataset.
        dim: Dimension to reduce along.
        group: Time component for grouped climatology: `'month'`, `'season'`, or `'year'`.
        **kwargs: Forwarded to `xr.Dataset.mean` (e.g. `skipna`, `keep_attrs`).

    Returns:
        Reduced dataset. Gains a `group` dimension when `group` is set.
    """
    return _groupby(data, group).mean(dim=dim, **kwargs)


@register_stat("max")
def max(data: xr.Dataset, *, dim: str = "time", group: str | None = None, **kwargs: Any) -> xr.Dataset:
    """Maximum value along a dimension.

    Wraps [`xr.Dataset.max`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.max.html).
    Any extra keyword arguments are forwarded to xarray.

    Args:
        data: Input dataset.
        dim: Dimension to reduce along.
        group: Time component for grouped climatology: `'month'`, `'season'`, or `'year'`.
        **kwargs: Forwarded to `xr.Dataset.max` (e.g. `skipna`, `keep_attrs`).

    Returns:
        Reduced dataset. Gains a `group` dimension when `group` is set.
    """
    return _groupby(data, group).max(dim=dim, **kwargs)


@register_stat("min")
def min(data: xr.Dataset, *, dim: str = "time", group: str | None = None, **kwargs: Any) -> xr.Dataset:
    """Minimum value along a dimension.

    Wraps [`xr.Dataset.min`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.min.html).
    Any extra keyword arguments are forwarded to xarray.

    Args:
        data: Input dataset.
        dim: Dimension to reduce along.
        group: Time component for grouped climatology: `'month'`, `'season'`, or `'year'`.
        **kwargs: Forwarded to `xr.Dataset.min` (e.g. `skipna`, `keep_attrs`).

    Returns:
        Reduced dataset. Gains a `group` dimension when `group` is set.
    """
    return _groupby(data, group).min(dim=dim, **kwargs)


@register_stat("std")
def std(data: xr.Dataset, *, dim: str = "time", group: str | None = None, **kwargs: Any) -> xr.Dataset:
    """Standard deviation along a dimension.

    Wraps [`xr.Dataset.std`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.std.html).
    Any extra keyword arguments are forwarded to xarray.

    Args:
        data: Input dataset.
        dim: Dimension to reduce along.
        group: Time component for grouped climatology: `'month'`, `'season'`, or `'year'`.
        **kwargs: Forwarded to `xr.Dataset.std` (e.g. `skipna`, `ddof`).

    Returns:
        Reduced dataset. Gains a `group` dimension when `group` is set.
    """
    return _groupby(data, group).std(dim=dim, **kwargs)


@register_stat("count")
def count(data: xr.Dataset, *, dim: str = "time", group: str | None = None, **kwargs: Any) -> xr.Dataset:
    """Count of non-NaN values along a dimension.

    Wraps [`xr.Dataset.count`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.count.html).
    Useful as a data-availability metric.

    Args:
        data: Input dataset.
        dim: Dimension to reduce along.
        group: Time component for grouped climatology: `'month'`, `'season'`, or `'year'`.
        **kwargs: Forwarded to `xr.Dataset.count`.

    Returns:
        Reduced dataset with integer counts. Gains a `group` dimension when `group` is set.
    """
    return _groupby(data, group).count(dim=dim, **kwargs)


@register_stat("quantile")
def quantile(
    data: xr.Dataset,
    *,
    dim: str = "time",
    group: str | None = None,
    q: list[float],
    **kwargs: Any,
) -> xr.Dataset:
    """Quantiles along a dimension.

    Wraps [`xr.Dataset.quantile`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.quantile.html).

    Note:
        Quantile computation requires the entire time axis to be in memory.
        Use `chunks: {time: -1}` together with `tiles` on the call to control
        peak memory usage on large grids. On large grids also set
        ``use_flox: false`` on the call — flox's quantile path uses roughly 2×
        the memory of the native xarray implementation.

    Args:
        data: Input dataset.
        dim: Dimension to reduce along.
        group: Time component for grouped climatology: `'month'`, `'season'`, or `'year'`.
        q: Quantile level(s) to compute, in [0, 1].
        **kwargs: Forwarded to `xr.Dataset.quantile` (e.g. `method`, `keep_attrs`).

    Returns:
        Reduced dataset with a `quantile` dimension. Gains a `group` dimension when `group` is set.
    """
    return _groupby(data, group).quantile(q=q, dim=dim, **kwargs)


@register_stat("pcount")
def pcount(data: xr.Dataset, *, dim: str = "time", **kwargs) -> xr.Dataset:
    """Percentage of non-NaN values along a dimension.

    Values are in [0, 100]. Useful for reporting data coverage.

    Args:
        data: Input dataset.
        dim: Dimension to reduce along.

    Returns:
        Dataset with values in [0, 100].
    """
    return 100 * data.count(dim) / data[dim].size


def _np_mode(
    arr: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
    bin_centres: np.ndarray,
) -> np.float32:
    """Return the centre of the most-occupied histogram bin.

    Returns NaN when all values are non-finite or the total weight is zero.
    Module-level so it is picklable for dask workers.
    """
    valid = np.isfinite(arr) & np.isfinite(weights)
    if not valid.any():
        return np.float32(np.nan)
    hist, _ = np.histogram(arr[valid], bins=bins, weights=weights[valid])
    if hist.sum() == 0:
        return np.float32(np.nan)
    return bin_centres[np.argmax(hist)].astype(np.float32)


def _apply_mode(
    arr: xr.DataArray,
    weights: xr.DataArray,
    dim: str,
    bins: np.ndarray,
    bin_centres: np.ndarray,
) -> xr.DataArray:
    return xr.apply_ufunc(
        _np_mode,
        arr,
        weights,
        kwargs={"bins": bins, "bin_centres": bin_centres},
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[]],
        exclude_dims={dim},
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )


@register_stat("mode")
def mode(
    data: xr.Dataset,
    *,
    dim: str = "time",
    group: str | None = None,
    bins: list[float],
    weight_var: str | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Mode (most frequent value) along a dimension, computed via a histogram.

    The return value is the centre of the most-occupied bin. For ordinal or
    discrete data (e.g. Douglas Sea Scale 0–9) pass half-integer bin edges so
    each integer gets its own bin:

    ```yaml
    bins: [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
    ```

    For continuous data, pass edges that capture the expected range at the
    desired resolution (e.g. Hs in 0.25 m increments).

    Args:
        data: Input dataset.
        dim: Dimension to reduce along (default ``'time'``).
        group: Time component for grouped climatology: ``'month'``, ``'season'``,
            or ``'year'``. When set, the output gains a dimension named after the
            group.
        bins: Bin edges as an explicit list. Must span the full data range —
            values outside the range are silently ignored by ``np.histogram``.
            Bin centres (midpoints of each edge pair) are returned as the mode
            value.
        weight_var: Variable in *data* to use as histogram weights. When
            ``None`` every sample contributes equally (occurrence mode). Pass
            ``'hs'`` for an Hs-weighted mode or ``'hs'`` squared for an
            energy-weighted mode.
        **kwargs: Accepted but not forwarded (pipeline compatibility).

    Returns:
        Reduced dataset with one variable per input variable (excluding
        *weight_var*). Gains a *group* dimension when *group* is set.

    Note:
        ``bins`` must be provided explicitly. Auto-detection from data range is
        not supported because it would force an eager compute on dask arrays.
    """
    bins_arr = np.asarray(bins, dtype="float64")
    bin_centres = (0.5 * (bins_arr[:-1] + bins_arr[1:])).astype("float32")

    compute_vars = [v for v in data.data_vars if v != weight_var]

    def _compute(ds: xr.Dataset) -> xr.Dataset:
        w = (
            ds[weight_var].astype("float32")
            if weight_var is not None
            else xr.ones_like(ds[compute_vars[0]], dtype="float32")
        )
        return xr.Dataset(
            {
                v: _apply_mode(ds[v].astype("float32"), w, dim, bins_arr, bin_centres)
                for v in compute_vars
            }
        )

    if group is not None:
        parts = [
            _compute(group_ds).expand_dims({group: [key]})
            for key, group_ds in data.groupby(f"{dim}.{group}")
        ]
        return xr.concat(parts, dim=group)

    return _compute(data)
