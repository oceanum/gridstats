"""Joint distribution functions."""
import logging
import numpy as np
import xarray as xr

from oncore.date import daterange, _parse
from onstats.utils import stepwise


logger = logging.getLogger(__name__)

_FILLVALUE = int(-2 ** 32 / 2)


def distribution3_timestep(
    self,
    dset,
    freq="1m",
    var1="hs",
    var2="tp",
    var3="dpm",
    bins1={"start": 0, "step": 0.5},
    bins2={"start": 0, "step": 1.0},
    bins3={"start": 0, "stop": 360, "step": 45},
    isdir1=False,
    isdir2=False,
    isdir3=True,
    dim="time",
    group="month",
    rechunk={},
    loadstep=True,
):
    """3D Joint distribution over timesteps to handle memory.

    Args:
        - self (instance): Instance argument required for plugging into Stats class.
        - dset (xr.Dataset): Dataset with variables to calculate distributions for.
        - var1 (str): Name of first variable in dset to compute joint distribution from.
        - var2 (str): Name of second variable in dset to compute joint distribution from.
        - var3 (str): Name of third variable in dset to compute joint distribution from.
        - bins1 (array, dict): Lower edges or arange kwargs to define bins for var1.
        - bins2 (array, dict): Lower edges or arange kwargs to define bins for var2.
        - bins3 (array, dict): Lower edges or arange kwargs to define bins for var3.
        - isdir1 (bool): True if var1 is a directional variable, False otherwise.
        - isdir2 (bool): True if var2 is a directional variable, False otherwise.
        - isdir3 (bool): True if var3 is a directional variable, False otherwise.
        - dim (str): Dimension to calculate distribution along.
        - group (str): Time grouping type, any valid time_{group} such as month, season.
        - rechunk (dict): Dimension: Size for rechunking each sliced time step.
        - loadstep (bool): Load each step before moving to the next one.

    Returns:
        - dsout (xr.Dataset): Dataset with 3D joint distribution.

    Note:
        - The bins args can be a list/array with lower bin edges or a dictionary with
          np.arange kwargs 'start', 'stop' and 'step' ('start' and 'stop' are estimated
          from the data range if not available as keys).
        - Mask is defined based on missing values from first variable (typically hs).

    """
    label = f"dist_{var1}_{var2}_{var3}"

    da1 = dset[var1]
    da2 = dset[var2]
    da3 = dset[var3]

    bin1_name = f"{var1}_bin"
    bin2_name = f"{var2}_bin"
    bin3_name = f"{var3}_bin"

    bins1 = _set_bins(bins1, da1)
    bins2 = _set_bins(bins2, da2)
    bins3 = _set_bins(bins3, da3)

    # Direction wrapping
    if isdir1:
        bins1 = bins1 - ((bins1[1] - bins1[0]) / 2)
        da1 = _wrap_directions(da1, dirmax=bins1.max())
    if isdir2:
        bins2 = bins2 - ((bins2[1] - bins2[0]) / 2)
        da2 = _wrap_directions(da2, dirmax=bins2.max())
    if isdir3:
        bins3 = bins3 - ((bins3[1] - bins3[0]) / 2)
        da3 = _wrap_directions(da3, dirmax=bins3.max())

    # Bin coordinates at cell centre
    coords = {
        bin1_name: bins1[:-1] + (bins1[1] - bins1[0]) / 2,
        bin2_name: bins2[:-1] + (bins2[1] - bins2[0]) / 2,
        bin3_name: bins3[:-1] + (bins3[1] - bins3[0]) / 2,
    }

    # Coordinates attributes
    attrs = {
        bin1_name: {
            "standard_name": f"{da1.attrs.get('standard_name', 'hs')}_bin",
            "long_name": f"{da1.attrs.get('long_name', 'hs')} bin",
            "units": da1.attrs.get("units", "m"),
        },
        bin2_name: {
            "standard_name": f"{da2.attrs.get('standard_name', 'tp')}_bin",
            "long_name": f"{da2.attrs.get('long_name', 'tp')} bin",
            "units": da2.attrs.get("units", "s"),
        },
        bin3_name: {
            "standard_name": f"{da3.attrs.get('standard_name', 'dp')}_bin",
            "long_name": f"{da3.attrs.get('long_name', 'dp')} bin",
            "units": da3.attrs.get("units", "degree"),
        },
    }

    # Looping based on time frequency
    alltimes = dset.time.to_index().to_pydatetime()
    times = list(daterange(start=alltimes[0], end=alltimes[-1], freq=freq))
    times.append(None)

    for ind, (t0, t1) in enumerate(zip(times[:-1], times[1:])):

        tslice = list(dset.time.sel(time=slice(t0, t1)).to_index().to_pydatetime())
        if tslice[-1] == t1:
            tslice.pop(-1)
        logger.info(f"Adding count for time step {ind + 1}/{len(times) - 1} ({tslice[0]} to {tslice[-1]})")

        da1t = da1.sel(time=tslice)
        da2t = da2.sel(time=tslice)
        da3t = da3.sel(time=tslice)
        if rechunk:
            logger.info(f"Rechunking time slice as {rechunk}")
            da1t = da1t.chunk(rechunk)
            da2t = da2t.chunk(rechunk)
            da3t = da3t.chunk(rechunk)

        # Grouping by
        if group is not None:
            logger.debug(f"Grouping by {group}")
            da1t = da1t.groupby(f"time.{group}")
            da2t = da2t.groupby(f"time.{group}")
            da3t = da3t.groupby(f"time.{group}")

        # Computing
        ds = xr.apply_ufunc(
            _np_histogram_3d,
            da1t,
            da2t,
            da3t,
            bins1,
            bins2,
            bins3,
            input_core_dims=[[dim], [dim], [dim], ["dummy1"], ["dummy2"], ["dummy3"]],
            output_core_dims=[[bin1_name, bin2_name, bin3_name]],
            exclude_dims=set((dim,)),
            vectorize=True,
            dask="parallelized",
            output_dtypes=["int32"],
            dask_gufunc_kwargs={
                "output_sizes": {
                    bin1_name: bins1.size - 1,
                    bin2_name: bins2.size - 1,
                    bin3_name: bins3.size - 1,
                },
            },
        )
        ds = ds.assign_coords(coords).to_dataset(name=label)
        if loadstep:
            ds = ds.load()

        if ind == 0:
            dsout = ds
        else:
            dsout = dsout.broadcast_like(ds).fillna(0.)
            dsout += ds.broadcast_like(dsout).fillna(0.)

    # Chunking output before saving to disk
    chunks = {**dict(dsout.dims), **{bin1_name: 1, bin2_name: 1, bin3_name: 1}}
    dsout = dsout.chunk(chunks)

    # Masking based on the first variable
    mask = da1.isel(**{dim: 0}, drop=True).notnull()
    mask = mask.chunk({dim: size for dim, size in zip(mask.dims, mask.shape)})
    dsout = dsout.where(mask)

    # Attributes
    dsout[label].attrs = {
        "standard_name": "data_count",
        "long_name": "number of valid data points",
        "units": "",
    }
    dsout[bin1_name].attrs = attrs[bin1_name]
    dsout[bin2_name].attrs = attrs[bin2_name]
    dsout[bin3_name].attrs = attrs[bin3_name]
    dsout[label].encoding = {"dtype": "int32", "_FillValue": _FILLVALUE}

    return dsout


def distribution3(
    self,
    dset,
    var1="hs",
    var2="tp",
    var3="dpm",
    bins1={"start": 0, "step": 0.5},
    bins2={"start": 0, "step": 1.0},
    bins3={"start": 0, "stop": 360, "step": 45},
    isdir1=False,
    isdir2=False,
    isdir3=True,
    dim="time",
    group="month",
):
    """3D Joint distribution.

    Args:
        - self (instance): Instance argument required for plugging into Stats class.
        - dset (xr.Dataset): Dataset with variables to calculate distributions for.
        - var1 (str): Name of first variable in dset to compute joint distribution from.
        - var2 (str): Name of second variable in dset to compute joint distribution from.
        - var3 (str): Name of third variable in dset to compute joint distribution from.
        - bins1 (array, dict): Lower edges or arange kwargs to define bins for var1.
        - bins2 (array, dict): Lower edges or arange kwargs to define bins for var2.
        - bins3 (array, dict): Lower edges or arange kwargs to define bins for var3.
        - isdir1 (bool): True if var1 is a directional variable, False otherwise.
        - isdir2 (bool): True if var2 is a directional variable, False otherwise.
        - isdir3 (bool): True if var3 is a directional variable, False otherwise.
        - dim (str): Dimension to calculate distribution along.
        - group (str): Time grouping type, any valid time_{group} such as month, season.

    Returns:
        - dsout (xr.Dataset): Dataset with 3D joint distribution.

    Note:
        - The bins args can be a list/array with lower bin edges or a dictionary with
          np.arange kwargs 'start', 'stop' and 'step' ('start' and 'stop' are estimated
          from the data range if not available as keys).
        - Mask is defined based on missing values from first variable (typically hs).

    """
    label = f"dist_{var1}_{var2}_{var3}"

    da1 = dset[var1]
    da2 = dset[var2]
    da3 = dset[var3]

    bin1_name = f"{var1}_bin"
    bin2_name = f"{var2}_bin"
    bin3_name = f"{var3}_bin"

    bins1 = _set_bins(bins1, da1)
    bins2 = _set_bins(bins2, da2)
    bins3 = _set_bins(bins3, da3)

    # Direction wrapping
    if isdir1:
        bins1 = bins1 - ((bins1[1] - bins1[0]) / 2)
        da1 = _wrap_directions(da1, dirmax=bins1.max())
    if isdir2:
        bins2 = bins2 - ((bins2[1] - bins2[0]) / 2)
        da2 = _wrap_directions(da2, dirmax=bins2.max())
    if isdir3:
        bins3 = bins3 - ((bins3[1] - bins3[0]) / 2)
        da3 = _wrap_directions(da3, dirmax=bins3.max())

    # Bin coordinates at cell centre
    coords = {
        bin1_name: bins1[:-1] + (bins1[1] - bins1[0]) / 2,
        bin2_name: bins2[:-1] + (bins2[1] - bins2[0]) / 2,
        bin3_name: bins3[:-1] + (bins3[1] - bins3[0]) / 2,
    }

    # Coordinates attributes
    attrs = {
        bin1_name: {
            "standard_name": f"{da1.attrs.get('standard_name', 'hs')}_bin",
            "long_name": f"{da1.attrs.get('long_name', 'hs')} bin",
            "units": da1.attrs.get("units", "m"),
        },
        bin2_name: {
            "standard_name": f"{da2.attrs.get('standard_name', 'tp')}_bin",
            "long_name": f"{da2.attrs.get('long_name', 'tp')} bin",
            "units": da2.attrs.get("units", "s"),
        },
        bin3_name: {
            "standard_name": f"{da3.attrs.get('standard_name', 'dp')}_bin",
            "long_name": f"{da3.attrs.get('long_name', 'dp')} bin",
            "units": da3.attrs.get("units", "degree"),
        },
    }

    # Mask based on the first variable
    mask = da1.isel(**{dim: 0}, drop=True).notnull()

    # Grouping by
    if group is not None:
        logger.debug(f"Grouping by {group}")
        da1 = da1.groupby(f"time.{group}")
        da2 = da2.groupby(f"time.{group}")
        da3 = da3.groupby(f"time.{group}")

    # Computing
    dsout = xr.apply_ufunc(
        _np_histogram_3d,
        da1,
        da2,
        da3,
        bins1,
        bins2,
        bins3,
        input_core_dims=[[dim], [dim], [dim], ["dummy1"], ["dummy2"], ["dummy3"]],
        output_core_dims=[[bin1_name, bin2_name, bin3_name]],
        exclude_dims=set((dim,)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=["int32"],
        dask_gufunc_kwargs={
            "output_sizes": {
                bin1_name: bins1.size - 1,
                bin2_name: bins2.size - 1,
                bin3_name: bins3.size - 1,
            },
        },
    )
    dsout = dsout.assign_coords(coords).where(mask).to_dataset(name=label)

    # Attributes
    dsout[label].attrs = {
        "standard_name": "data_count",
        "long_name": "number of valid data points",
        "units": "",
    }
    dsout[bin1_name].attrs = attrs[bin1_name]
    dsout[bin2_name].attrs = attrs[bin2_name]
    dsout[bin3_name].attrs = attrs[bin3_name]
    dsout[label].encoding = {"dtype": "int32", "_FillValue": _FILLVALUE}

    return dsout


def distribution2(
    self,
    dset,
    var1="wspd",
    var2="wdir",
    bins1={"start": 0, "step": 0.5},
    bins2={"start": 0, "stop": 360, "step": 45},
    isdir1=False,
    isdir2=True,
    dim="time",
    group="month",
):
    """2D joint distribution.

    Args:
        - self (instance): Instance argument required for plugging into Stats class.
        - dset (xr.Dataset): Dataset with variables to calculate distributions for.
        - var1 (str): Name of first variable in dset to compute joint distribution from.
        - var2 (str): Name of second variable in dset to compute joint distribution from.
        - bins1 (array, dict): Lower edges or arange kwargs to define bins for var1.
        - bins2 (array, dict): Lower edges or arange kwargs to define bins for var2.
        - isdir1 (bool): True if var1 is a directional variable, False otherwise.
        - isdir2 (bool): True if var2 is a directional variable, False otherwise.
        - dim (str): Dimension to calculate distribution along.
        - group (str): Time grouping type, any valid time_{group} such as month, season.

    Returns:
        - dsout (xr.Dataset): Dataset with 2D joint distribution.

    Note:
        - The bins args can be a list/array with lower bin edges or a dictionary with
          np.arange kwargs 'start', 'stop' and 'step' ('start' and 'stop' are estimated
          from the data range if not available as keys).
        - Mask is defined based on missing values from first variable (typically wspd).

    """
    label = f"dist_{var1}_{var2}"

    da1 = dset[var1]
    da2 = dset[var2]

    bin1_name = f"{var1}_bin"
    bin2_name = f"{var2}_bin"

    bins1 = _set_bins(bins1, da1)
    bins2 = _set_bins(bins2, da2)

    # Direction wrapping
    if isdir1:
        bins1 = bins1 - ((bins1[1] - bins1[0]) / 2)
        da1 = _wrap_directions(da1, dirmax=bins1.max())
    if isdir2:
        bins2 = bins2 - ((bins2[1] - bins2[0]) / 2)
        da2 = _wrap_directions(da2, dirmax=bins2.max())

    # Bin coordinates at cell centre
    coords = {
        bin1_name: bins1[:-1] + (bins1[1] - bins1[0]) / 2,
        bin2_name: bins2[:-1] + (bins2[1] - bins2[0]) / 2,
    }

    # Coordinates attributes
    attrs = {
        bin1_name: {
            "standard_name": f"{da1.attrs.get('standard_name', 'wspd')}_bin",
            "long_name": f"{da1.attrs.get('long_name', 'wspd')} bin",
            "units": da1.attrs.get("units", "m/s"),
        },
        bin2_name: {
            "standard_name": f"{da2.attrs.get('standard_name', 'wdir')}_bin",
            "long_name": f"{da2.attrs.get('long_name', 'wdir')} bin",
            "units": da2.attrs.get("units", "degree"),
        },
    }

    # Mask based on the first variable
    mask = da1.isel(**{dim: 0}, drop=True).notnull()

    # Grouping by
    if group is not None:
        logger.debug(f"Grouping by {group}")
        da1 = da1.groupby(f"time.{group}")
        da2 = da2.groupby(f"time.{group}")

    # Computing
    dsout = xr.apply_ufunc(
        _np_histogram_2d,
        da1,
        da2,
        bins1,
        bins2,
        input_core_dims=[[dim], [dim], ["dummy1"], ["dummy2"]],
        output_core_dims=[[bin1_name, bin2_name]],
        exclude_dims=set((dim,)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=["int32"],
        dask_gufunc_kwargs={
            "output_sizes": {
                bin1_name: bins1.size - 1,
                bin2_name: bins2.size - 1,
            },
        },
    )
    dsout = dsout.assign_coords(coords).where(mask).to_dataset(name=label)

    # Attributes
    dsout[label].attrs = {
        "standard_name": "data_count",
        "long_name": "number of valid data points",
        "units": "",
    }
    dsout[bin1_name].attrs = attrs[bin1_name]
    dsout[bin2_name].attrs = attrs[bin2_name]
    dsout[label].encoding = {"dtype": "int32", "_FillValue": _FILLVALUE}

    return dsout


def _set_bins(bins, darr):
    """Construct bins from different arguments options."""
    if isinstance(bins, (list, np.ndarray)):
        return np.array(bins)
    elif isinstance(bins, dict):
        try:
            step = bins.get("step")
        except KeyError:
            raise ValueError(
                f"'step' is required when specifying bins as arange kwargs ({bins})"
            )
        if "start" not in bins:
            vmin = float(darr.min())
            bins["start"] = vmin - (vmin % step)
        if "stop" not in bins:
            vmax = float(darr.max())
            bins["stop"] = vmax + (vmax % step)
        logger.debug(f"Constructing bins from arange kwargs: {bins}")
        return np.arange(**bins)
    else:
        raise ValueError(
            f"bins must be a list/array or a dict specifying arange kwargs, got {bins}"
        )


def _wrap_directions(darr, dirmax):
    """Wrap directions DataArray."""
    return xr.where(darr <= dirmax, darr, darr - 360)


def _np_histogram_2d(arr1, arr2, bins1, bins2):
    """Histogram 2D.

    Args:
        - arr1 (1d array): First array to compute histogram from, e.g., Wspd.
        - arr2 (1d array): Second array to compute histogram from, e.g., Wdir.
        - bins1 (1d array): Bin edges for arr1.
        - bins2 (1d array): Bin edges for arr2.

    Returns:
        - The multidimensional histogram of (arr1, arr2).

    """
    dist, __, __ = np.histogram2d(x=arr1, y=arr2, bins=(bins1, bins2), density=False)
    return dist


def _np_histogram_3d(arr1, arr2, arr3, bins1, bins2, bins3):
    """Histogram 3D.

    Args:
        - arr1 (1d array): First array to compute histogram from, e.g., Hs.
        - arr2 (1d array): Second array to compute histogram from, e.g., Tp.
        - arr3 (1d array): Third array to compute histogram from, e.g., Dp.
        - bins1 (1d array): Bin edges for arr1.
        - bins2 (1d array): Bin edges for arr2.
        - bins3 (1d array): Bin edges for arr3.

    Returns:
        - The multidimensional histogram of (arr1, arr2, arr3).

    """
    dist, __ = np.histogramdd(
        sample=(arr1, arr2, arr3), bins=(bins1, bins2, bins3), density=False
    )
    return dist


if __name__ == "__main__":
    from onstats.utils import uv_to_spddir

    dset = xr.open_dataset(
        "/source/onhindcast/implementation/swan/tasman/model/tasman-19790201T00-grid.nc",
        chunks={},
    )
    dset["wspd"], dset["wdir"] = uv_to_spddir(dset.xwnd, dset.ywnd, coming_from=True)

    # dsout = distribution3(
    #     None,
    #     dset,
    #     var1="hs",
    #     var2="tps",
    #     var3="dpm",
    #     bins1={"step": 1.0},
    #     bins2=[0, 5, 10, 15, 20],
    #     bins3={"step": 90.0},
    #     dim="time",
    #     group="month",
    # )

    dsout = distribution2(
        None,
        dset,
        var1="wspd",
        var2="wdir",
        bins1={"start": 0, "step": 1.0},
        bins2={"start": 0, "stop": 360, "step": 45},
        dim="time",
        group="month",
    )
