"""Core stats functions."""
import logging
import numpy as np
import dask.array as da
import pandas as pd
import xarray as xr
from scipy import stats, signal


logger = logging.getLogger(__name__)


def _timestep(df):
    """Timestep from regularly-spaced dataframe with time-based index.

    Args:
        df (pd.Dataframe, pd.Series): Pandas object with a time-based index.

    Returns:
        dt (timedelta): Regular time-step in Timedelta format.

    """
    if isinstance(df, (xr.Dataset, xr.DataArray)):
        tdiff = np.diff(df.time)
    elif isinstance(df, (pd.DataFrame, pd.Series)):
        tdiff = np.diff(df.index)
    if tdiff.min() != tdiff.max():
        raise ValueError("Times are not regularly-spaced in time")
    return pd.to_timedelta(tdiff[0])


def _pov(data, dt_hour, percentile=95, duration=24):
    """Peaks over threshold.

    Args:
        data (1d array): Timeseries data to select peaks from.
        percentile (float): Percentile above which peaks are selected.
        duration (float): Hours in storm below which extra peaks are discarded.

    Return:
        peaks (pd.Series): Subset pandas series with data for selected peaks.
        height (float): Threshold value above which peaks are defined.

    """
    distance = duration / dt_hour
    ind_perc = int(0.01 * percentile * data.shape[0])
    height = np.sort(data)[ind_perc]
    ind = signal.find_peaks(data, height=height, distance=distance)[0]
    return data[ind], height


def nrpv(
    data,
    dt_hour,
    return_periods=[1, 5, 10, 20, 50, 100, 1000, 10000],
    percentile=95,
    distribution="gumbel_r",
    duration=24,
):
    """Return period values from numpy array.

    Args:
        data (1d array): Data to calculate rpv for.
        dt_hour(float): Time step in data.
        return_periods (list): Return period years to calculate rpv values for.
        percentile (float): Percentile above which peaks are selected.
        distribution (str): Statistical distribution to fit the data, any valid
            distribution in scipy.stats, e.g., "gumbel_r", "weibull_min", etc.
        duration (float): Hours in storm below which extra peaks are discarded.

    Returns:
        rpvs (dict): Return period values for years in return_periods.

    """
    # Ensure valid distrubution
    try:
        func = getattr(stats, distribution)
    except AttributeError as err:
        raise ValueError(
            f"Distribution '{distribution}' not available in scipy.stats, valid "
            f"distributions are: {[f for f in dir(stats) if f[0].islower()]}"
        ) from err

    # Ensure no missing values
    if np.isnan(data).any():
        logger.debug("Missing values not allowed, returning nan")
        return np.tile(np.nan, len(return_periods))

    peaks, height = _pov(data, dt_hour, percentile, duration)

    if peaks.size == 0:
        logger.debug(f"No peaks over {height} ({percentile}th percentile), returning nan")
        return np.tile(np.nan, len(return_periods))

    fits = func.fit(peaks, floc=height)
    dt_year = dt_hour / (24 * 365)
    ntimes = data.shape[0]
    npeaks = peaks.shape[0]
    rpvs = []
    for return_period in return_periods:
        p = ntimes * dt_year / (return_period * npeaks)
        rpvs.append(func.isf(p, *fits))
    return np.array(rpvs)


def xrpv(
    darr,
    return_periods=[1, 5, 10, 20, 50, 100, 1000, 10000],
    percentile=95,
    distribution="gumbel_r",
    duration=24,
):
    """Return period values from DataArray.

    Args:
        darr (xr.DataArray): Data to calculate rpv for.
        return_periods (list): Return period years to calculate rpv values for.
        percentile (float): Percentile above which peaks are selected.
        distribution (str): Statistical distribution to fit the data, any valid
            distribution in scipy.stats, e.g., "gumbel_r", "weibull_min", etc.
        duration (float): Hours in storm below which extra peaks are discarded.

    Returns:
        rpvs (dict): Return period values for years in return_periods.

    """
    dt_hour = _timestep(darr).total_seconds() / 3600
    dsout = xr.apply_ufunc(
        nrpv,
        darr,
        dt_hour,
        return_periods,
        percentile,
        distribution,
        duration,
        input_core_dims=[["time"], [], ["period"], [], [], []],
        output_core_dims=[["period"]],
        exclude_dims=set(("time",)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"]
    )
    dsout = dsout.assign_coords({"period": return_periods})
    dsout.period.attrs = {
        "standard_name": "return_period",
        "long_name": "return period",
        "units": "year",
    }
    return dsout.transpose("period", ...)


if __name__ == "__main__":

    import datetime
    dset = xr.open_dataset(
        "/source/onhindcast/implementation/swan/jogchum/useast/model/grid/useast-20000501T00-grid.nc"
    )
    darr = dset[["hs", "tps"]]#.isel(latitude=[0,1,2])#, longitude=-1)
    darr = darr.chunk({"longitude": None, "latitude": None, "time": None})
    then = datetime.datetime.now()
    ret = xrpv(darr)
    elapsed = datetime.datetime.now() - then
    print(f"Elapsed time: {round(elapsed.total_seconds())} sec")

