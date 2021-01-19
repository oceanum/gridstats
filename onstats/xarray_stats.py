"""Core stats functions operating on xarray objects."""
import logging
import numpy as np
import dask.array as da
import pandas as pd
import xarray as xr

from onstats.numpy_stats import np_rpv


logger = logging.getLogger(__name__)


def _timestep(df, dim="time"):
    """Timestep from regularly-spaced dataframe with time-based index.

    Args:
        df (pd.Dataframe, pd.Series): Pandas object with a time-based index.
        dim (str): Time dimension if xarray object.

    Returns:
        dt (timedelta): Regular time-step in Timedelta format.

    """
    if isinstance(df, (xr.Dataset, xr.DataArray)):
        tdiff = np.diff(df[dim])
    elif isinstance(df, (pd.DataFrame, pd.Series)):
        tdiff = np.diff(df.index)
    if tdiff.min() != tdiff.max():
        raise ValueError("Times are not regularly-spaced in time")
    return pd.to_timedelta(tdiff[0])


def rpv(
    darr,
    return_periods=[1, 5, 10, 20, 50, 100, 1000, 10000],
    percentile=95,
    distribution="gumbel_r",
    duration=24,
    dim="time",
):
    """Return period values from DataArray.

    Args:
        darr (xr.DataArray): Data to calculate rpv for.
        return_periods (list): Return period years to calculate rpv values for.
        percentile (float): Percentile above which peaks are selected.
        distribution (str): Statistical distribution to fit the data, any valid
            distribution in scipy.stats, e.g., "gumbel_r", "weibull_min", etc.
        duration (float): Hours in storm below which extra peaks are discarded.
        dim (str): Dimension to calculate rpv along.

    Returns:
        rpvs (dict): Return period values for years in return_periods.

    """
    dt_hour = _timestep(darr, dim).total_seconds() / 3600
    dsout = xr.apply_ufunc(
        np_rpv,
        darr,
        dt_hour,
        return_periods,
        percentile,
        distribution,
        duration,
        input_core_dims=[[dim], [], ["period"], [], [], []],
        output_core_dims=[["period"]],
        exclude_dims=set((dim,)),
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
    return dsout.transpose("period", ...).chunk({"period": 1})


if __name__ == "__main__":

    import datetime
    from dask.diagnostics.progress import ProgressBar

    dset = xr.open_dataset(
        "/source/onhindcast/implementation/swan/jogchum/useast/model/grid/useast-20000501T00-grid.nc"
    )
    darr = dset[["hs", "tps"]]#.isel(latitude=[0,1,2])#, longitude=-1)
    darr = darr.chunk({"longitude": None, "latitude": None, "time": None})
    then = datetime.datetime.now()
    ret = rpv(darr)
    with ProgressBar():
        elapsed = datetime.datetime.now() - then
        ret = ret.load()
    print(f"Elapsed time: {round(elapsed.total_seconds())} sec")
    ret.to_netcdf("/home/rguedes/tmp/rpv.nc")

