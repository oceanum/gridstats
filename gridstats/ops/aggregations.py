"""Standard xarray aggregation operations."""
from __future__ import annotations

from typing import Any

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
