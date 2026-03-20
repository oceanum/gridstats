"""Return Period Value (extreme value) operations."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import signal, stats as scipy_stats
import xarray as xr

from onstats.registry import register_stat

logger = logging.getLogger(__name__)

_DEFAULT_RETURN_PERIODS = [1, 5, 10, 20, 50, 100, 1000, 10000]


def _pot(
    data: np.ndarray, dt_hours: float, percentile: float, duration: float
) -> np.ndarray:
    """Extract peaks over threshold using the percentile method.

    Args:
        data: 1-D data array.
        dt_hours: Timestep in hours.
        percentile: Percentile used to define the threshold.
        duration: Minimum time (hours) between peaks.

    Returns:
        Array of peak values above the threshold.
    """
    threshold = np.nanpercentile(data, percentile)
    min_distance = max(1, int(duration / dt_hours))
    peaks, _ = signal.find_peaks(data, height=threshold, distance=min_distance)
    return data[peaks]


def _np_rpv(
    data: np.ndarray,
    dt_hours: float,
    return_periods: list[float],
    percentile: float,
    duration: float,
    distribution: str,
) -> np.ndarray:
    """Fit an extreme-value distribution and return values at given return periods.

    Args:
        data: 1-D data array (may contain NaNs).
        dt_hours: Timestep in hours.
        return_periods: Return periods in years.
        percentile: Percentile for peaks-over-threshold selection.
        duration: Minimum peak separation in hours.
        distribution: scipy.stats distribution name (e.g. 'gumbel_r').

    Returns:
        Array of shape (len(return_periods),) with estimated values.
    """
    nans = np.full(len(return_periods), np.nan, dtype="float32")
    data = data[~np.isnan(data)]
    if data.size < 2:
        return nans

    peaks = _pot(data, dt_hours, percentile, duration)
    if peaks.size < 2:
        return nans

    dist = getattr(scipy_stats, distribution)
    try:
        params = dist.fit(peaks)
    except Exception:
        return nans

    # Convert return period (years) → exceedance probability
    n_years = (data.size * dt_hours) / 8760.0
    n_peaks_per_year = peaks.size / max(n_years, 1e-9)
    exceedance_prob = 1.0 / (np.array(return_periods) * n_peaks_per_year)
    exceedance_prob = np.clip(exceedance_prob, 1e-9, 1.0)

    return dist.isf(exceedance_prob, *params).astype("float32")


@register_stat("rpv")
def rpv(
    data: xr.Dataset,
    *,
    dim: str = "time",
    return_periods: list[float] = _DEFAULT_RETURN_PERIODS,
    percentile: float = 95,
    distribution: str = "gumbel_r",
    duration: float = 24,
    group: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Compute return period values using extreme value analysis.

    Args:
        data: Input dataset.
        dim: Time dimension name.
        return_periods: Return periods in years at which to estimate values.
        percentile: Percentile threshold for peaks-over-threshold selection.
        distribution: scipy.stats continuous distribution name.
        duration: Minimum time (hours) between detected peaks.
        group: Time component to group by before computing RPV
            (e.g. 'month'). Not commonly used for RPV.

    Returns:
        Dataset with a 'period' dimension containing the estimated values.

    Raises:
        ValueError: If the distribution is not available in scipy.stats or
            if the timestep exceeds the specified duration.
    """
    if not hasattr(scipy_stats, distribution):
        raise ValueError(
            f"Distribution '{distribution}' not found in scipy.stats."
        )

    dt_ns = float(np.diff(data[dim].values[:2])[0])
    dt_hours = dt_ns / 3.6e12
    if dt_hours > duration:
        raise ValueError(
            f"Timestep ({dt_hours:.1f}h) is larger than duration ({duration}h). "
            "Peaks-over-threshold requires dt < duration."
        )

    dsout = xr.apply_ufunc(
        _np_rpv,
        data,
        kwargs={
            "dt_hours": dt_hours,
            "return_periods": return_periods,
            "percentile": percentile,
            "duration": duration,
            "distribution": distribution,
        },
        input_core_dims=[[dim]],
        output_core_dims=[["period"]],
        exclude_dims={dim},
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"],
        dask_gufunc_kwargs={
            "output_sizes": {"period": len(return_periods)},
            "allow_rechunk": True,
        },
    )
    dsout = dsout.assign_coords({"period": return_periods})
    dsout["period"].attrs = {
        "standard_name": "return_period",
        "long_name": "return period",
        "units": "year",
    }
    return dsout
