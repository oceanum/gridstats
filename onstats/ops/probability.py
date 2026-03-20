"""Range-based probability operations."""
from __future__ import annotations

import logging

import numpy as np
import dask.array as da
import xarray as xr

from onstats.registry import register_stat

logger = logging.getLogger(__name__)


@register_stat("range_probability")
def range_probability(
    data: xr.Dataset,
    *,
    dim: str = "time",
    data_ranges: list[dict],
    **kwargs,
) -> xr.Dataset:
    """Compute the probability that each variable falls within specified ranges.

    Args:
        data: Input dataset.
        dim: Dimension along which to compute probabilities.
        data_ranges: List of range specifications, each a dict with keys:

            - ``var`` (str): Variable name in ``data``.
            - ``start`` (float | None): Lower bound (None = no lower bound).
            - ``stop`` (float | None): Upper bound (None = no upper bound).
            - ``left`` ('closed' | 'open'): Whether the lower bound is inclusive.
            - ``right`` ('closed' | 'open'): Whether the upper bound is inclusive.
            - ``label`` (str, optional): Output variable name. Defaults to
              ``{var}_{start}_to_{stop}``.

    Returns:
        Dataset with one probability variable per range specification.
    """
    if not isinstance(data_ranges, list):
        data_ranges = [data_ranges]

    left_funcs = {"closed": da.greater_equal, "open": da.greater}
    right_funcs = {"closed": da.less_equal, "open": da.less}

    dsout = xr.Dataset()
    for spec in data_ranges:
        dvar = spec["var"]
        darr = data[dvar]

        start = spec["start"] if spec.get("start") is not None else -np.inf
        stop = spec["stop"] if spec.get("stop") is not None else np.inf

        lfunc = left_funcs[spec.get("left", "closed")]
        rfunc = right_funcs[spec.get("right", "closed")]

        llabel = f"{start:g}" if spec.get("start") is not None else "min"
        rlabel = f"{stop:g}" if spec.get("stop") is not None else "max"
        varname = spec.get("label", f"{dvar}_{llabel}_to_{rlabel}")

        in_range = lfunc(darr, start) & rfunc(darr, stop)
        dsout[varname] = in_range.sum(dim=dim) / darr.count(dim)

    return dsout
