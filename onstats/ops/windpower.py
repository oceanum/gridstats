"""Wind turbine power estimation."""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
import xarray as xr

from onstats.registry import register_stat


def _pol3(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Third-order polynomial power curve."""
    return a * x + b * x**2 + c * x**3 + d


@register_stat("winpow")
def winpow(
    data: xr.Dataset,
    *,
    dim: str = "time",
    turbine_power: float,
    cutin: float = 3.0,
    rated: float = 10.61,
    cutout: float = 25.0,
    agg: str | None = "mean",
    group: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Estimate wind turbine power from wind speed.

    Fits a third-order polynomial to a reference power curve, then applies
    cut-in, rated, and cut-out speed thresholds.

    Args:
        data: Input dataset. Must contain a wind-speed variable (first data var
            is used as the wind-speed input).
        dim: Time dimension name.
        turbine_power: Rated mechanical power output (kW).
        cutin: Cut-in wind speed below which output is zero (m/s).
        rated: Rated wind speed above which output is capped (m/s).
        cutout: Cut-out wind speed above which output is zero (m/s).
        agg: Aggregation to apply after computing instantaneous power
            ('mean', 'max', etc.). Pass None to return the full time series.
        group: Time component to group by (only supported when agg is set).

    Returns:
        Dataset with wind power variable(s).

    Raises:
        NotImplementedError: If group is provided without agg.
    """
    if group is not None and agg is None:
        raise NotImplementedError("groupby requires agg to be set.")

    # Reference power curve fitted with polynomial
    x_ref = np.array([3.0, 3.6, 4.32, 5.06, 6.2, 7.05, 8.83, 10.61])
    y_ref = np.array([1500, 1915, 2719, 3872, 6110, 7963, 13490, 16000])
    fits, _ = curve_fit(_pol3, x_ref, y_ref)

    # Compute power for each wind-speed variable
    dsout = xr.Dataset()
    for varname, wspd in data.data_vars.items():
        power = _pol3(wspd, *fits)
        power = power.where(wspd > cutin, 0)
        power = power.where(wspd < rated, turbine_power)
        power = power.where(wspd < cutout, 0)

        if agg is not None:
            if group is not None:
                power = getattr(power.groupby(f"{dim}.{group}"), agg)(dim=dim)
            else:
                power = getattr(power, agg)(dim=dim)

        dsout[varname] = power

    return dsout
