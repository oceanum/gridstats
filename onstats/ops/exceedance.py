"""Exceedance and non-exceedance probability operations."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import signal
import xarray as xr

from onstats.registry import register_stat

logger = logging.getLogger(__name__)


def _duration_to_hours(duration: str) -> float:
    """Convert a duration string (e.g. '3h', '1d') to hours."""
    td = pd.to_timedelta(duration)
    return td.total_seconds() / 3600


def _values_above_duration(
    data: np.ndarray, dt_hours: float, durations: list[float]
) -> np.ndarray:
    """Fraction of time that True values in data persist for at least each duration.

    Args:
        data: Boolean 1-D array.
        dt_hours: Timestep in hours.
        durations: List of minimum duration thresholds in hours.

    Returns:
        Array of shape (len(durations),) with fractions in [0, 1].
    """
    if not data.any():
        return np.zeros(len(durations))

    # Find runs of True values and their lengths
    peaks, props = signal.find_peaks(
        data.astype(float), plateau_size=(1, data.size)
    )
    plateau_sizes = props["plateau_sizes"] * dt_hours  # convert steps → hours

    fractions = np.empty(len(durations))
    for i, dur in enumerate(durations):
        steps_meeting_duration = np.sum(
            plateau_sizes[plateau_sizes >= dur] / dt_hours
        )
        fractions[i] = steps_meeting_duration / data.size

    return fractions


def _probability_of_occurrence(
    data: xr.Dataset,
    condition: xr.Dataset,
    duration: str | list[str],
    dim: str,
    group: str | None,
) -> xr.Dataset:
    """Compute probability that condition holds for at least `duration`.

    Args:
        data: Original dataset (used for timestep calculation).
        condition: Boolean dataset (same shape as data).
        duration: Duration threshold string(s) e.g. '0h', '1d', ['1d', '7d'].
        dim: Time dimension name.
        group: Optional time grouping ('month', 'season', 'year').

    Returns:
        Dataset with probabilities per duration.
    """
    durations = [duration] if isinstance(duration, str) else duration
    duration_hours = [_duration_to_hours(d) for d in durations]

    # Timestep in hours
    dt_ns = float(np.diff(data[dim].values[:2])[0])
    dt_hours = dt_ns / 3.6e12  # ns → hours

    results = []
    for dur_str, dur_h in zip(durations, duration_hours):
        if dur_h == 0:
            # Simple mean (no duration filtering)
            if group is not None:
                prob = condition.groupby(f"{dim}.{group}").mean(dim=dim)
            else:
                prob = condition.mean(dim=dim)
        else:
            prob = xr.apply_ufunc(
                lambda x: _values_above_duration(x.astype(bool), dt_hours, [dur_h])[0],
                condition,
                input_core_dims=[[dim]],
                vectorize=True,
                dask="parallelized",
                output_dtypes=["float32"],
            )
            if group is not None:
                prob = prob.expand_dims(
                    {group: condition.groupby(f"{dim}.{group}").groups.keys()}
                )

        if len(durations) > 1:
            prob = prob.expand_dims({"duration": [dur_str]})
        results.append(prob)

    if len(durations) > 1:
        return xr.concat(results, dim="duration")
    return results[0]


@register_stat("exceedance")
def exceedance(
    data: xr.Dataset,
    *,
    dim: str = "time",
    threshold: float,
    maxval: float = np.inf,
    inclusive: bool = True,
    duration: str | list[str] = "0h",
    group: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Probability of exceedance above a threshold.

    Args:
        data: Input dataset.
        dim: Time dimension name.
        threshold: Lower bound for exceedance.
        maxval: Upper bound (default: no upper bound).
        inclusive: Whether the threshold comparison is inclusive (>=).
        duration: Minimum duration the condition must persist. Accepts a string
            like '3h' or '1d', or a list of strings for multiple durations.
        group: Time component to group by (e.g. 'month', 'year').

    Returns:
        Dataset with exceedance probabilities. Variable names are suffixed
        with the threshold value.
    """
    cmp = data.__ge__ if inclusive else data.__gt__
    condition = cmp(threshold) & (data <= maxval)
    dsout = _probability_of_occurrence(data, condition, duration, dim, group)
    return dsout.rename({v: f"{v}_{threshold:g}" for v in dsout.data_vars})


@register_stat("nonexceedance")
def nonexceedance(
    data: xr.Dataset,
    *,
    dim: str = "time",
    threshold: float,
    inclusive: bool = True,
    duration: str | list[str] = "0h",
    group: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Probability of non-exceedance below a threshold.

    Args:
        data: Input dataset.
        dim: Time dimension name.
        threshold: Upper bound for non-exceedance.
        inclusive: Whether the comparison is inclusive (<=).
        duration: Minimum duration the condition must persist.
        group: Time component to group by (e.g. 'month', 'year').

    Returns:
        Dataset with non-exceedance probabilities. Variable names are suffixed
        with the threshold value.
    """
    cmp = data.__le__ if inclusive else data.__lt__
    condition = cmp(threshold)
    dsout = _probability_of_occurrence(data, condition, duration, dim, group)
    return dsout.rename({v: f"{v}_{threshold:g}" for v in dsout.data_vars})
