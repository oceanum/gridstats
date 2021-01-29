"""Core stats functions operating on xarray objects."""
import logging
import numpy as np
import dask.array as da
import pandas as pd
import xarray as xr
from itertools import product

from onstats.numpy_stats import np_rpv, wave_histogram


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


def _wrap_directions(darr, dirmax):
    """Wrap directions DataArray."""
    return xr.where(darr <= dirmax, darr, darr - 360)


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


def distribution(
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
        hs (xr.DataArray): Significant wave height data.
        tp (xr.DataArray): Peak wave period data.
        dp (xr.DataArray): Peak wave direction data.
        hs_bins (1d array): Bin edges for Hs.
        tp_bins (1d array): Bin edges for Tp.
        dp_bins (1d array): Bin edges for Dp.
        dim (str): Dimension to calculate distribution along.
        label (str): Name for joint distribution variable.

    Returns:
        Dataset with Hs, Tp, Dp joint distributio along dim.

    """
    hs_bins = np.array(hs_bins)
    tp_bins = np.array(tp_bins)
    dp_bins = np.array(dp_bins)

    # Direction wrapping
    dp_bins = dp_bins - ((dp_bins[1] - dp_bins[0]) / 2)
    if isinstance(dp, xr.core.groupby.DataArrayGroupBy):
        grouping = f"time.{dp._unique_coord.name}"
        dp = dp.map(_wrap_directions, dirmax=dp_bins.max())
        # When a function is applied the groupby object is stacked
        dp = dp.groupby(grouping)
    else:
        dp = _wrap_directions(dp, dirmax=dp_bins.max())

    # Bin coordinates at cell centre
    coords = {
        "hs_bin": hs_bins[:-1] + (hs_bins[1] - hs_bins[0]) / 2,
        "tp_bin": tp_bins[:-1] + (tp_bins[1] - tp_bins[0]) / 2,
        "dp_bin": dp_bins[:-1] + (dp_bins[1] - dp_bins[0]) / 2,
    }

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

    # Masking based on Hs
    logger.debug(f"Masking land from output")
    if isinstance(hs, xr.core.groupby.DataArrayGroupBy):
        for g, hsi in hs:
            break
        for g, tpi in tp:
            break
        for g, dpi in dp:
            break
    else:
        hsi = hs
        tpi = tp
        dpi = dp
    dsout = dsout.where(hsi.isel(**{dim: 0}, drop=True).notnull())

    # Attributes
    dsout[label].attrs = {
        "standard_name": "data_count",
        "long_name": "number of valid data points",
        "units": "",
    }
    dsout.hs_bin.attrs = {
        "standard_name": f"{hsi.attrs.get('standard_name', 'hs')}_bin",
        "long_name": f"{hsi.attrs.get('long_name', 'hs')} bin",
        "units": "m"
    }
    dsout.tp_bin.attrs = {
        "standard_name": f"{tpi.attrs.get('standard_name', 'tp')}_bin",
        "long_name": f"{tpi.attrs.get('long_name', 'tp')} bin",
        "units": "s"
    }
    dsout.dp_bin.attrs = {
        "standard_name": f"{dpi.attrs.get('standard_name', 'dp')}_bin",
        "long_name": f"{dpi.attrs.get('long_name', 'dp')} bin",
        "units": "degree"
    }

    dsout[label].encoding = {"dtype": "int32", "_FillValue": -32767}

    return dsout


def directional_stat(
    dset,
    func,
    dir_var,
    nsector=4,
    dim="time",
    **kwargs,
):
    """Bin per directional sectors and apply xarray function.

    Args:
        dset (xr.Dataset): Xarray dataset to calculate stats over.
        func (str): Name of valid xarray function to apply.
        dir_var (str): Directional data var to bin data over.
        nsector (int): Number of directional sectors.
        dim (str): Dimension to apply function over.
        kwargs: Kwargs for function func.

    """
    # Binning data per directional sector
    dirs = dset[dir_var]
    dsector = 360 / nsector
    sectors = np.linspace(0, 360 - dsector, nsector)
    starts = (sectors - dsector / 2) % 360
    stops = (sectors + dsector / 2) % 360
    dsout = []
    for start, stop in zip(starts, stops):
        if stop > start:
            mask = (dirs >= start) & (dirs < stop)
        else:
            mask = (dirs >= start) | (dirs < stop)
        dsout.append(dset.where(mask))

    # Concat directional bins into new dimension
    dsout = xr.concat(dsout, dim="direction").assign_coords({"direction": sectors})
    dsout["direction"].attrs = {
        "standard_name": dirs.attrs.get("standard_name", "direction"),
        "long_name": dirs.attrs.get("standard_name", "direction sector"),
        "units": dirs.attrs.get("units", "degree"),
        "variable_name": dir_var,
    }

    # Calculate dask stats
    dsout = getattr(dsout, func)(dim=dim, **kwargs)

    return dsout


if __name__ == "__main__":

    import datetime
    from dask.diagnostics.progress import ProgressBar

    dset = xr.open_zarr("/data/ww3/ww3_grid_sample.zarr", consolidated=True)
    dset["tp"] = 1 / dset.fp

    #====================
    # Histogram approach
    #====================
    ranges = {
        "hs": {"start": 0, "end": 20, "freq": 0.5},
        "tp": {"start": 0, "end": 20, "freq": 1},
        "dp": {"start": 0, "end": 360, "freq": 45},
    }

    # ds = dset[["hs","tp","dp"]].isel(latitude=0, longitude=0).load()
    ds = dset[["hs","tp","dp"]].load()

    hs_bins = np.arange(
        ranges["hs"]["start"],
        ranges["hs"]["end"]+ranges["hs"]["freq"],
        ranges["hs"]["freq"]
    )
    tp_bins = np.arange(
        ranges["tp"]["start"],
        ranges["tp"]["end"]+ranges["tp"]["freq"],
        ranges["tp"]["freq"]
    )
    dp_bins = np.arange(
        ranges["dp"]["start"],
        ranges["dp"]["end"]+ranges["dp"]["freq"],
        ranges["dp"]["freq"]
    )

    hs = ds.hs.groupby("time.month")
    tp = ds.tp.groupby("time.month")
    dp = ds.dp.groupby("time.month")
    # dsg = ds.groupby("time.month")

    dsout = distribution(hs, tp, dp, hs_bins, tp_bins, dp_bins)

    with ProgressBar():
        dsout = dsout.load()


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
