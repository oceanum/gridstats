"""Core stats functions operating on xarray objects."""
import logging
import numpy as np
import dask.array as da
import pandas as pd
import xarray as xr
from itertools import product

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
    elif isinstance(df, xr.core.groupby.DatasetGroupBy):
        tdiff = np.diff(list(df)[0][1][dim])
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
        output_dtypes=["float32"],
    )
    dsout = dsout.assign_coords({"period": return_periods})
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


def distribution(dset, ranges, dim="time", mask_var=None, mapping={}):
    """Distribution statistics.

    Args:
        dset (xr.Dataset): Dataset to calculate distribution from.
        ranges (dict): Bins definition, keys are the variable names, values
            are the kwargs for pandas.interval_range function to define bins,
            e.g. ranges={"hs": {"start": 0, "end": 3, "freq": 0.5}, "tp": {"start": 0, "end": 20, "freq": 5}}.
        dim (str): Dimension to calculate distribution along.
        mask_var (str): Name of variable to use for masking land.
        mapping (dict): Mapping to rename distribution variables.

    """
    data_vars = list(ranges.keys())
    missing_vars = list(set(data_vars) - set(dset.data_vars))
    if missing_vars:
        raise KeyError(f"Vars {missing_vars} defined in ranges not in dset {dset}")

    ds = dset[data_vars].rename(mapping)
    data_vars = list(ds.data_vars)
    nvar = len(data_vars)
    ax = [i - nvar for i in range(nvar)]

    # Combinations of bins to loop over
    interval_ranges = [pd.interval_range(**kwargs) for kwargs in ranges.values()]
    ranges_iterator = product(*iter(interval_ranges))

    # Summing data along dim within each bin combination
    dsout = []
    for data_ranges in ranges_iterator:
        logger.debug(f"Data Ranges: {data_ranges}")
        coords = {}
        mask = True
        for data_var, data_range in zip(data_vars, data_ranges):
            coords.update({f"{data_var}_bin": [data_range.mid]})
            mask *= (ds[data_var] >= data_range.left) & (ds[data_var] < data_range.right)
        dsout.append(ds.where(mask).count(dim).expand_dims(dim=coords, axis=ax))
    dsout = xr.combine_by_coords(dsout)

    # Total count based on first variable in ranges dict
    dsout["data_count"] = ds[data_vars[0]].count(dim)

    # Masking
    if mask_var is not None:
        mask = dset[mask_var]
        if dim in mask.dims:
            mask = mask.isel(**{dim: 0})
        mask = mask.notnull()
        logger.debug(f"Masking output from {mask}")
        dsout = dsout.where(mask)

    # Attributes
    for data_var in data_vars:
        # Data variable
        dsout[f"{data_var}"].attrs["standard_name"] = (
            ds[f"{data_var}"].attrs.get("standard_name", data_var) + "_count"
        )
        dsout[f"{data_var}"].attrs["long_name"] = (
            ds[f"{data_var}"].attrs.get("long_name", data_var) + " count"
        )
        dsout[f"{data_var}"].attrs["units"] = ""
        dsout["data_count"].attrs = {
            "standard_name": "data_count",
            "long_name": "number of valid data points",
            "units": "",
        }
        # Coordinate
        dsout[f"{data_var}_bin"].attrs["standard_name"] = (
            ds[f"{data_var}"].attrs.get("standard_name", data_var) + "_bin"
        )
        dsout[f"{data_var}_bin"].attrs["long_name"] = (
            ds[f"{data_var}"].attrs.get("long_name", data_var) + " bin"
        )
        dsout[f"{data_var}_bin"].attrs["units"] = ds[f"{data_var}"].attrs.get(
            "units", data_var
        )

    return dsout


if __name__ == "__main__":

    import datetime
    from dask.diagnostics.progress import ProgressBar

    dset = xr.open_dataset(
        "/source/onhindcast/implementation/swan/jogchum/useast/model/grid/useast-20000501T00-grid.nc"
    ).chunk()

    ranges = {
        "hs": {"start": 0, "end": 5, "freq": 1},
        "tps": {"start": 0, "end": 20, "freq": 5},
        "dpm": {"start": 0, "end": 360, "freq": 90},
    }
    dsout = distribution(
        dset=dset,
        ranges={
            "hs": {"start": 0, "end": 5, "freq": 1},
            "tps": {"start": 0, "end": 20, "freq": 5},
            "dpm": {"start": 0, "end": 360, "freq": 90},
        },
        mask_var="hs",
        mapping={"tps": "tp", "dpm": "dp"},
    )

    # darr = dset[["hs", "tps"]]  # .isel(latitude=[0,1,2])#, longitude=-1)
    # darr = darr.chunk({"longitude": None, "latitude": None, "time": None})
    # then = datetime.datetime.now()
    # ret = rpv(darr)
    # with ProgressBar():
    #     elapsed = datetime.datetime.now() - then
    #     ret = ret.load()
    # print(f"Elapsed time: {round(elapsed.total_seconds())} sec")
    # ret.to_netcdf("/home/rguedes/tmp/rpv.nc")

    # ds = dset[["hs"]].chunk({})
    # ret = rpv(ds.groupby("time.month"))
