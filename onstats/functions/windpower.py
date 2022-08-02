"""Wind Power."""
import numpy as np
from scipy.optimize import curve_fit

from onstats.functions.xarray_wrapper import _groupby


def _pol3(x, a, b, c, d):
    return (a * x) + (b * x ** 2) + (c * x ** 3) + d


def winpow(
    self,
    wspd,
    turbine_power,
    cutin=3.0,
    rated=10.61,
    cutout=25.0,
    agg="mean",
    dim="time",
    group=None,
):
    """Empirical turbine wind power.

    Args:
        - self (instance): Instance argument required for plugging into Stats class.
        - wspd (DataArray): Wind speed at the turbine elevation.
        - turbine_power (float): Rotor mechanical power (kW).
        - cutin (float): Cut-in wind speed below which power is zero.
        - rated (float): Rated wind speed above which wind power is optimised.
        - cutout (float): Cut-out wind speed above which wind power is zero.
        - agg (str): Aggregation function to apply.
        - dim (str): Dimension to calculate rpv along.
        - group (str): Time grouping type, any valid time_{group} such month, season.

    Returns:
        - winpow (xr.DataArray): Wind Power DataArray.

    Note:
        - Windpower should be a derived variable instead a function.

    """
    if group is not None:
        raise NotImplementedError("groupby not implemented with winpow function.")

    x = np.array([3.0, 3.6, 4.32, 5.06, 6.2, 7.05, 8.83, 10.61])
    y = np.array([1500, 1915, 2719, 3872, 6110, 7963, 13490, 16000])

    # Empirical fit
    fits, __ = curve_fit(_pol3, x, y)
    power = _pol3(wspd, *fits)

    # Zero below cut-in
    power = power.where(wspd > cutin, 0)

    # Maximum between rated and cut-out
    power = power.where(wspd < rated, turbine_power)

    # Zero above cut-out
    power = power.where(wspd < cutout, 0)

    # Aggregation
    if agg is not None:
        power = getattr(power, agg)(dim=dim)

    return power


if __name__ == "__main__":
    import xarray as xr
    dset = xr.open_dataset(
        "/data/forecast/ww3/glob-20220125T18.zarr",
        engine="zarr",
        chunks={},
    )
    wspd = np.sqrt(dset.uwnd**2 + dset.vwnd**2)
    dsout = winpow(None, wspd, turbine_power=15000)
