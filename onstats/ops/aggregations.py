"""Standard xarray aggregation operations."""
from __future__ import annotations

import xarray as xr

from onstats.registry import register_stat


def _groupby(
    data: xr.Dataset, group: str | None
) -> xr.Dataset | xr.core.groupby.DatasetGroupBy:
    """Apply groupby on the time dimension if group is provided."""
    if group is not None:
        return data.groupby(f"time.{group}")
    return data


@register_stat("mean")
def mean(data: xr.Dataset, *, dim: str = "time", group: str | None = None, **kwargs) -> xr.Dataset:
    """Compute the mean along dim, optionally grouped by a time component.

    Args:
        data: Input dataset.
        dim: Dimension to reduce along.
        group: Time component to group by (e.g. 'month', 'season', 'year').

    Returns:
        Reduced dataset.
    """
    return _groupby(data, group).mean(dim=dim, **kwargs)


@register_stat("max")
def max(data: xr.Dataset, *, dim: str = "time", group: str | None = None, **kwargs) -> xr.Dataset:
    """Compute the maximum along dim, optionally grouped by a time component.

    Args:
        data: Input dataset.
        dim: Dimension to reduce along.
        group: Time component to group by (e.g. 'month', 'season', 'year').

    Returns:
        Reduced dataset.
    """
    return _groupby(data, group).max(dim=dim, **kwargs)


@register_stat("min")
def min(data: xr.Dataset, *, dim: str = "time", group: str | None = None, **kwargs) -> xr.Dataset:
    """Compute the minimum along dim, optionally grouped by a time component.

    Args:
        data: Input dataset.
        dim: Dimension to reduce along.
        group: Time component to group by (e.g. 'month', 'season', 'year').

    Returns:
        Reduced dataset.
    """
    return _groupby(data, group).min(dim=dim, **kwargs)


@register_stat("std")
def std(data: xr.Dataset, *, dim: str = "time", group: str | None = None, **kwargs) -> xr.Dataset:
    """Compute the standard deviation along dim, optionally grouped by a time component.

    Args:
        data: Input dataset.
        dim: Dimension to reduce along.
        group: Time component to group by (e.g. 'month', 'season', 'year').

    Returns:
        Reduced dataset.
    """
    return _groupby(data, group).std(dim=dim, **kwargs)


@register_stat("count")
def count(data: xr.Dataset, *, dim: str = "time", group: str | None = None, **kwargs) -> xr.Dataset:
    """Count non-NaN values along dim, optionally grouped by a time component.

    Args:
        data: Input dataset.
        dim: Dimension to reduce along.
        group: Time component to group by (e.g. 'month', 'season', 'year').

    Returns:
        Reduced dataset.
    """
    return _groupby(data, group).count(dim=dim, **kwargs)


@register_stat("quantile")
def quantile(
    data: xr.Dataset,
    *,
    dim: str = "time",
    group: str | None = None,
    q: list[float],
    **kwargs,
) -> xr.Dataset:
    """Compute quantiles along dim, optionally grouped by a time component.

    Args:
        data: Input dataset.
        dim: Dimension to reduce along.
        group: Time component to group by (e.g. 'month', 'season', 'year').
        q: Quantile(s) to compute, in [0, 1].

    Returns:
        Reduced dataset with a 'quantile' dimension.
    """
    return _groupby(data, group).quantile(q=q, dim=dim, **kwargs)


@register_stat("pcount")
def pcount(data: xr.Dataset, *, dim: str = "time", **kwargs) -> xr.Dataset:
    """Compute the percentage of non-NaN values along dim.

    Args:
        data: Input dataset.
        dim: Dimension to reduce along.

    Returns:
        Dataset with values in [0, 100].
    """
    return 100 * data.count(dim) / data[dim].size
