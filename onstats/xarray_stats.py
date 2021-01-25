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
        # Ignore extra times in grouped for now because they jump from year to year
        tdiff = tdiff[[0]]
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


def wave_histogram(hs, tp, dp, hs_bins, tp_bins, dp_bins):
    """Multidimension wave histogram.

    Args:
        hs (1d array): Significant wave height array.
        Tp (1d array): Peak wave period array.
        Dp (1d array): Peak wave direction array.
        hs_bins (1d array): Bin edges for Hs.
        tp_bins (1d array): Bin edges for Tp.
        dp_bins (1d array): Bin edges for Dp.

    Returns:
        The multidimensional histogram of (Hs, Tp, Dp).

    """
    dist, __ = np.histogramdd(
        sample=(hs, tp, dp),
        bins=(hs_bins, tp_bins, dp_bins),
        normed=False
    )
    return dist


def fastdist(
    hs,
    tp,
    dp,
    hs_bins,
    tp_bins,
    dp_bins,
    dim="time",
    label="hs_tp_dp_dist",
):
    """Fast distribution statistics.

    Args:
        ranges (dict): Bins definition, keys are the variable names, values
            are the kwargs for pandas.interval_range function to define bins,
            e.g. ranges={"hs": {"start": 0, "stop": 3, "step": 0.5}, "tp": ...}.
        dim (str): Dimension to calculate distribution along.

    """
    hs_bins = np.array(hs_bins)
    tp_bins = np.array(tp_bins)
    dp_bins = np.array(dp_bins)

    # Bin coordinates at cell centre
    coords = {
        "hs_bin": hs_bins[:-1] + (hs_bins[1] - hs_bins[0]) / 2,
        "tp_bin": tp_bins[:-1] + (tp_bins[1] - tp_bins[0]) / 2,
        "dp_bin": dp_bins[:-1] + (dp_bins[1] - dp_bins[0]) / 2,
    }

    # data_vars = list(ranges.keys())
    # missing_vars = list(set(data_vars) - set(dset.data_vars))
    # if missing_vars:
    #     raise KeyError(f"Vars {missing_vars} from ranges not in dataset {dset}")

    # bins = [np.hstack((np.arange(**k), k["stop"])) for k in ranges.values()]

    dsout = xr.apply_ufunc(
        wave_histogram,
        hs,
        tp,
        dp,
        hs_bins,
        tp_bins,
        dp_bins,
        input_core_dims=[[dim], [dim], [dim], ["dummy1"], ["dummy2"], ["dummy3"]],
        output_core_dims=[["hs_bin", "tp_bin", "dp_bin"]],
        exclude_dims=set((dim,)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=["int32"],
        dask_gufunc_kwargs={
            # "allow_rechunking": True,
            "output_sizes": {
                "hs_bin": hs_bins.size - 1,
                "tp_bin": tp_bins.size - 1,
                "dp_bin": dp_bins.size - 1
            },
        },
    ).assign_coords(coords).to_dataset(name=label)

    # Total count based on Hs
    dsout["data_count"] = hs.count(dim)

    # Masking based on Hs
    logger.debug(f"Masking land from output")
    dsout = dsout.where(hs.isel(**{dim: 0}, drop=True).notnull())

    # Attributes
    dsout[label].attrs = {
        "standard_name": "data_count",
        "long_name": "number of valid data points",
        "units": "",
    }
    dsout["data_count"].attrs = {
        "standard_name": "data_count",
        "long_name": "number of valid data points",
        "units": "",
    }
    dsout.hs_bin.attrs = {
        "standard_name": f"{hs.attrs.get('standard_name', 'hs')}_bin",
        "long_name": f"{hs.attrs.get('long_name', 'hs')} bin",
        "units": "m"
    }
    dsout.tp_bin.attrs = {
        "standard_name": f"{tp.attrs.get('standard_name', 'tp')}_bin",
        "long_name": f"{tp.attrs.get('long_name', 'tp')} bin",
        "units": "m"
    }
    dsout.dp_bin.attrs = {
        "standard_name": f"{dp.attrs.get('standard_name', 'dp')}_bin",
        "long_name": f"{dp.attrs.get('long_name', 'dp')} bin",
        "units": "m"
    }

    return dsout


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

    Note: groupby dimensions can be included by calling groupby.map, e.g.
        dset.groupby("time.month").map(distribution, **distribution_kwargs).

    """
    data_vars = list(ranges.keys())
    missing_vars = list(set(data_vars) - set(dset.data_vars))
    if missing_vars:
        raise KeyError(f"Vars {missing_vars} defined in ranges not in dset {dset}")

    ds = dset[data_vars].rename(mapping)
    data_vars = list(ds.data_vars)
    nvar = len(data_vars)
    ax = [i - nvar for i in range(nvar)]

    label = "_".join(data_vars) + "_dist"
    ds0 = ds[[data_vars[0]]].rename({data_vars[0]: label})

    # Combinations of bins to loop over
    interval_ranges = [pd.interval_range(**kwargs) for kwargs in ranges.values()]
    ranges_iterator = product(*iter(interval_ranges))

    # Summing data along dim within each bin combination
    dsout = []
    for data_ranges in ranges_iterator:
        logger.debug(f"Data Ranges: {data_ranges}")
        coords = {}
        mask = True
        for datavar, datarange in zip(data_vars, data_ranges):
            coords.update({f"{datavar}_bin": [datarange.mid]})
            mask *= (ds[datavar] >= datarange.left) & (ds[datavar] < datarange.right)
        dsout.append(ds0.where(mask).count(dim).expand_dims(dim=coords, axis=ax))
    dsout = xr.combine_by_coords(dsout)

    # Total count based on first variable in ranges dict
    dsout["data_count"] = ds[data_vars[0]].count(dim)

    # Masking
    if mask_var is not None:
        mask = dset[mask_var]
        if dim in mask.dims:
            mask = mask.isel(**{dim: 0}, drop=True)
        mask = mask.notnull()
        logger.debug(f"Masking output from {mask}")
        dsout = dsout.where(mask)

    # Attributes
    dsout[label].attrs = {
        "standard_name": "data_count",
        "long_name": "number of valid data points",
        "units": "",
    }
    dsout["data_count"].attrs = {
        "standard_name": "data_count",
        "long_name": "number of valid data points",
        "units": "",
    }
    for data_var in data_vars:
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


    # Where masking approach
    print("Old method")
    ranges = {
        "hs": {"start": 0, "end": 5, "freq": 0.5},
        "tps": {"start": 0, "end": 20, "freq": 1},
        "dpm": {"start": 0, "end": 360, "freq": 45},
    }
    mapping = {"tps": "tp", "dpm": "dp"}
    dsout0 = distribution(
        dset=dset,
        ranges=ranges,
        mask_var="hs",
        mapping=mapping,
    )

    # dsout_month = dset.groupby("time.month").map(distribution, ranges=ranges, mask_var="hs", mapping=mapping)

    with ProgressBar():
        dsout0 = dsout0.load()


    #====================
    # Histogram approach
    #====================
    print("New method")
    ds = dset[["hs","tps","dpm"]]#.load()
    hs_bins = np.arange(
        ranges["hs"]["start"],
        ranges["hs"]["end"]+ranges["hs"]["freq"],
        ranges["hs"]["freq"]
    )
    tp_bins = np.arange(
        ranges["tps"]["start"],
        ranges["tps"]["end"]+ranges["tps"]["freq"],
        ranges["tps"]["freq"]
    )
    dp_bins = np.arange(
        ranges["dpm"]["start"],
        ranges["dpm"]["end"]+ranges["dpm"]["freq"],
        ranges["dpm"]["freq"]
    )
    dsout1 = fastdist(ds.hs, ds.tps, ds.dpm, hs_bins, tp_bins, dp_bins)

    with ProgressBar():
        dsout1 = dsout1.load()


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
