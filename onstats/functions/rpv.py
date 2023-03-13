"""Return Period Value function."""
import logging
import numpy as np
import dask.array as da
import pandas as pd
import xarray as xr
from scipy import stats, signal

from onstats.utils import get_timestep


logger = logging.getLogger(__name__)


def rpv(
    self,
    dset,
    return_periods=[1, 5, 10, 20, 50, 100, 1000, 10000],
    percentile=95,
    distribution="gumbel_r",
    duration=24,
    dim="time",
    group=None,
):
    """Return period values.

    Args:
        - self (instance): Instance argument required for plugging into Stats class.
        - dset (xr.Dataset): Dataset with variables to calculate rpv for.
        - return_periods (list): Return period years to calculate rpv values for.
        - percentile (float): Percentile above which peaks are selected.
        - distribution (str): Statistical distribution to fit the data, any valid
          distribution in scipy.stats, e.g., "gumbel_r", "weibull_min", etc.
        - duration (float): Hours in storm below which extra peaks are discarded.
        - dim (str): Dimension to calculate rpv along.
        - group (str): Time grouping type, any valid time_{group} such month, season.

    Returns:
        - rpvs (xr.Dataset): Return period values dataset.

    """
    # Distribution function
    try:
        func = getattr(stats, distribution)
    except AttributeError as err:
        raise ValueError(
            f"Distribution '{distribution}' not available in scipy.stats, valid "
            f"distributions are: {[f for f in dir(stats) if f[0].islower()]}"
        ) from err

    # Data time resolution
    dt_hour = get_timestep(dset, dim).total_seconds() / 3600
    if duration < dt_hour:
        raise ValueError(f"dt {dt_hour}h must be less than storm duration {duration}h")

    # Land mask
    mask = dset.isel(**{dim: -1}, drop=True).notnull()

    # Grouping by
    if group is not None:
        logger.debug(f"Grouping by {group}")
        dset = dset.groupby(f"time.{group}")

    # Calculate rpv for variables in dataset
    dsout = xr.apply_ufunc(
        _np_rpv,
        dset,
        func,
        dt_hour,
        return_periods,
        percentile,
        duration,
        input_core_dims=[[dim], [], [], ["period"], [], []],
        output_core_dims=[["period"]],
        exclude_dims=set((dim,)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"],
        dask_gufunc_kwargs={"allow_rechunk": True},
    ).where(mask)

    # Assign return period coordinate
    dsout = dsout.assign_coords({"period": return_periods})
    # Set attributes
    dsout.period.attrs = {
        "standard_name": "return_period",
        "long_name": "return period",
        "units": "year",
    }
    dsout.attrs = {
        "distribution": distribution,
        "threshold_percentile": percentile,
        "storm_duration": duration,
    }
    return dsout.transpose("period", ...).chunk({"period": 1})


def _pot(data, dt_hour, percentile=95, duration=24):
    """Peaks over threshold.

    Args:
        - data (1d array): Timeseries data to select peaks from.
        - percentile (float): Percentile above which peaks are selected.
        - duration (float): Hours in storm below which extra peaks are discarded.

    Return:
        - peaks (1d array): Timeseries data to find peaks from.
        - height (float): Threshold value above which peaks are defined.

    """
    distance = duration / dt_hour
    ind_perc = int(0.01 * percentile * data.shape[0])
    height = np.sort(data)[ind_perc]
    data = data.copy() # https://github.com/pydata/xarray/issues/3715
    ind = signal.find_peaks(data, height=height, distance=distance)[0]
    return data[ind], height


def _np_rpv(
    data,
    func,
    dt_hour,
    return_periods=[1, 5, 10, 20, 50, 100, 1000, 10000],
    percentile=95,
    duration=24,
):
    """Return period values from numpy array.

    Args:
        - data (1d array): Data to calculate rpv for.
        - func (obj): Statistical distribution for fitting the data, any valid
          distribution function in scipy.stats, e.g., `scipy.stats.gumbel_r`.
        - dt_hour(float): Time step in data.
        - return_periods (list): Return period years to calculate rpv values for.
        - percentile (float): Percentile above which peaks are selected.
        - duration (float): Hours in storm below which extra peaks are discarded.

    Returns:
        - rpvs (1d array): Return period values for years in return_periods.

    """
    peaks, height = _pot(data, dt_hour, percentile, duration)
    rpvs = [np.nan] * len(return_periods)
    if peaks.size > 0:
        try:
            fits = func.fit(peaks, floc=height)
        except (TypeError, AttributeError):
            logger.error(
                "TypeError or AttributeError when calling func.fit with "
                f"peaks={peaks} ({type(peaks)}), floc={height} ({type(height)})"
            )
            return np.array(rpvs)
        dt_year = dt_hour / (24 * 365)
        ntimes = data.shape[0]
        npeaks = peaks.shape[0]
        rpvs = []
        for return_period in return_periods:
            p = ntimes * dt_year / (return_period * npeaks)
            rpvs.append(func.isf(p, *fits))
    return np.array(rpvs)


if __name__ == "__main__":
    dset = xr.open_dataset("/source/onhindcast/implementation/swan/tasman/model/tasman-19790201T00-grid.nc")
    ds = rpv(None, dset[["hs"]], ystep=None, xstep=None, yname="latitude")#, xname="longitude")
