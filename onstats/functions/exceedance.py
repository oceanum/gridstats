"""Return Period Value function."""
import logging
import numpy as np
import dask.array as da
import pandas as pd
import xarray as xr
import datetime
from scipy import stats, signal

from oncore.date import timedelta
from onstats.utils import get_timestep


logger = logging.getLogger(__name__)


def exceedance(
    self,
    dset,
    threshold,
    maxval=np.inf,
    inclusive=True,
    duration="0h",
    dim="time",
    group=None,
):
    """Probability of exceedance over duration period.

    Args:
        - self (instance): Instance argument required for plugging into Stats class.
        - dset (Dataset, DataArray): Data to calculate exceedance for.
        - threshold (float): Threshold value at or above which exceedance is computed.
        - maxval (float): Compute exceedance for data below this value.
        - inclusive (bool): If True the threshold and maxval values are included.
        - duration (str, list): Duration below which exceeding values are discarded,
          use a list for exceedances over different durations.
        - dim (str): Dimension to calculate exceedance along.
        - group (str): Time grouping type, any valid time_{group} such month, season.

    Returns:
        - dsout (Dataset, DataArray): Probability of exceedance.

    """
    # Exceedance array
    if inclusive:
        condition = (dset >= threshold) & (dset <= maxval)
    else:
        condition = (dset > threshold) & (dset < maxval)
    ds = xr.where(condition, 1, 0)

    dsout = _probability_of_occurrance(ds, duration=duration, dim=dim, group=group)

    # Land masking
    mask = dset.isel(**{dim: -1}).notnull()
    dsout = dsout.where(mask)

    # Set attributes
    dsout.duration.attrs = {
        "standard_name": "exceedance_duration",
        "long_name": "duration above which exceedance is computed",
        "units": "",
    }
    dsout.attrs = {
        "standard_name": "exceedance_probability",
        "long_name": "probability of exceedance at or above duration period",
        "units": "",
    }

    return dsout


def nonexceedance(
    self,
    dset,
    threshold,
    inclusive=True,
    duration="0h",
    dim="time",
    group=None,
):
    """Probability of non-exceedance over duration period.

    Args:
        - self (instance): Instance argument required for plugging into Stats class.
        - dset (Dataset, DataArray): Data to calculate non-exceedance for.
        - threshold (float): Threshold value at or above which non-exceedance is computed.
        - inclusive (bool): If True the threshold value is included.
        - duration (str, list): Duration below which non-exceeding values are,
          discarded, use a list for non-exceedances over different durations.
        - dim (str): Dimension to calculate non-exceedance along.
        - group (str): Time grouping type, any valid time_{group} such month, season.

    Returns:
        - dsout (Dataset, DataArray): Probability of non-exceedance.

    """
    # Exceedance array
    if inclusive:
        ds = xr.where(dset <= threshold, 1, 0)
    else:
        ds = xr.where(dset < threshold, 1, 0)

    dsout = _probability_of_occurrance(ds, duration=duration, dim=dim, group=group)

    # Land masking
    mask = dset.isel(**{dim: -1}).notnull()
    dsout = dsout.where(mask)

    # Set attributes
    dsout.duration.attrs = {
        "standard_name": "non_exceedance_duration",
        "long_name": "duration above which non exceedance is computed",
        "units": "",
    }
    dsout.attrs = {
        "standard_name": "non_exceedance_probability",
        "long_name": "probability of non exceedance at or above duration period",
        "units": "",
    }

    return dsout


def _probability_of_occurrance(dset, duration, dim="time", group=None):
    """Probability of occurrance over duration period.

    Args:
        - dset (Dataset, DataArray): Binary data indicating occurrance (1) or not (0).
        - duration (str, list): Duration below which true values are discarded, use a
          list for probabilities over different durations.
        - dim (str): Dimension to calculate probability of occurrance along.
        - group (str): Time grouping type, any valid time_{group} such month, season.

    Returns:
        - dsout (xr.Dataset): Probability of occurrance dataset.

    """
    # Dataset time resolution
    dt_hour = get_timestep(dset, dim).total_seconds() / 3600

    # Durations in hours
    duration = duration if isinstance(duration, list) else [duration]
    t0 = datetime.datetime(1970, 1, 1)
    duration_hour = []
    for dt in duration:
        duration_hour.append((t0 + timedelta(dt) - t0).total_seconds() / 3600)

    # Grouping by
    if group is not None:
        logger.debug(f"Grouping by {group}")
        dset = dset.groupby(f"time.{group}")

    # Calculate occurrances for the full data
    dsout = xr.apply_ufunc(
        _values_over_threshold,
        dset,
        dt_hour,
        duration_hour,
        input_core_dims=[[dim], [], ["duration"]],
        output_core_dims=[["duration"]],
        exclude_dims=set((dim,)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"],
    )

    # Scale by count
    dsout = dsout / dset.count(dim)

    # Assign duration coordinate
    dsout = dsout.assign_coords({"duration": duration})

    return dsout.transpose("duration", ...).chunk({"duration": 1})


def _values_over_threshold(data, dt, durations):
    """Values over threshold during or above required duration.

    Args:
        - data (1d array): Binary timeseries indicating exceedance.
        - dt (array): Time step of data in hours.
        - durations (list): Hours in storm below which extra peaks are discarded.

    Return:
        - vot (1d array): Values over threshold corresponding to each duration.

    """
    vot = []
    for duration in durations:
        # Distance corresponding to duration
        if duration < dt:
            raise ValueError(f"dt {dt}h must be less than storm duration {duration}h")
        distance = round(duration / dt)

        # Append zeros at start and end to ensure boundary values are detected, this would
        # be more efficient at xarray level function but that results in multiple time chunks
        data = np.hstack([0, data, 0])
        ind, prop = signal.find_peaks(data, height=1.0, plateau_size=distance)
        vot.append(prop["plateau_sizes"].sum())

    return da.from_array(vot, chunks=(1,))


if __name__ == "__main__":
    dset = xr.open_dataset("/data/forecast/glob-20211214T12.nc", chunks={})
    dset = dset.hs.isel(latitude=slice(None, None, 10), longitude=slice(None, None, 10))#.sel(longitude=173, latitude=-40, drop=True)
    ds1 = exceedance(None, dset, threshold=3.0, duration=["3h", "6h", "9h"]).load()
    # ds2 = nonexceedance(None, dset, threshold=3.0, duration=["3h", "6h", "9h"]).load()
