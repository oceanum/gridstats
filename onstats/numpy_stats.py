"""Core stats functions operating on numpy arrays to use with xarray.apply_ufuncs."""
import logging
import numpy as np
import dask.array as da
import pandas as pd
from scipy import stats, signal


logger = logging.getLogger(__name__)


def _pov(data, dt_hour, percentile=95, duration=24):
    """Peaks over threshold.

    Args:
        data (1d array): Timeseries data to select peaks from.
        percentile (float): Percentile above which peaks are selected.
        duration (float): Hours in storm below which extra peaks are discarded.

    Return:
        peaks (1d array): Timeseries data to find peaks from.
        height (float): Threshold value above which peaks are defined.

    """
    if duration < dt_hour:
        raise ValueError(
            f"timestep({dt_hour}h) must be less than the storm duration ({duration}h)"
        )
    distance = duration / dt_hour
    ind_perc = int(0.01 * percentile * data.shape[0])
    height = np.sort(data)[ind_perc]
    ind = signal.find_peaks(data, height=height, distance=distance)[0]
    return data[ind], height


def np_rpv(
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
        rpvs (1d array): Return period values for years in return_periods.

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
        return da.from_array([np.nan] * len(return_periods), chunks=(1,))

    peaks, height = _pov(data, dt_hour, percentile, duration)

    if peaks.size == 0:
        logger.debug(
            f"No peaks over {height} ({percentile}th percentile), returning nan"
        )
        return da.from_array([np.nan] * len(return_periods), chunks=(1,))

    fits = func.fit(peaks, floc=height)
    dt_year = dt_hour / (24 * 365)
    ntimes = data.shape[0]
    npeaks = peaks.shape[0]
    rpvs = []
    for return_period in return_periods:
        p = ntimes * dt_year / (return_period * npeaks)
        rpvs.append(func.isf(p, *fits))
    return da.from_array(rpvs, chunks=(1,))


def np_histogram_2d(arr1, arr2, bins1, bins2):
    """Histogram 2D.

    Args:
        arr1 (1d array): First array to compute histogram from, e.g., Wspd.
        arr2 (1d array): Second array to compute histogram from, e.g., Wdir.
        bins1 (1d array): Bin edges for arr1.
        bins2 (1d array): Bin edges for arr2.

    Returns:
        The multidimensional histogram of (arr1, arr2).

    """
    dist, __, __ = np.histogram2d(x=arr1, y=arr2, bins=(bins1, bins2), normed=False)
    return dist


def np_histogram_3d(arr1, arr2, arr3, bins1, bins2, bins3):
    """Histogram 3D.

    Args:
        arr1 (1d array): First array to compute histogram from, e.g., Hs.
        arr2 (1d array): Second array to compute histogram from, e.g., Tp.
        arr3 (1d array): Third array to compute histogram from, e.g., Dp.
        bins1 (1d array): Bin edges for arr1.
        bins2 (1d array): Bin edges for arr2.
        bins3 (1d array): Bin edges for arr3.

    Returns:
        The multidimensional histogram of (arr1, arr2, arr3).

    """
    dist, __ = np.histogramdd(
        sample=(arr1, arr2, arr3), bins=(bins1, bins2, bins3), normed=False
    )
    return dist
