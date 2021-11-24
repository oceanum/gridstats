"""Joint distribution functions."""
import logging
import numpy as np
import xarray as xr


logger = logging.getLogger(__name__)


def _set_bins(bins, darr):
    """Construct bins from different arguments options."""
    if isinstance(bins, (list, np.ndarray)):
        return np.array(bins)
    elif isinstance(bins, dict):
        logger.info(f"Constructing bins from arange kwargs: {bins}")
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
        return np.arange(**bins)
    else:
        raise ValueError(
            f"bins must be a list/array or a dict specifying arange kwargs, got {bins}"
        )


def distribution(
    self,
    dset,
    var1,
    var2,
    var3,
    bins1,
    bins2,
    bins3,
    dim="time",
    group="month",
):
    """Joint wave distribution (Hs, Tp, Dp).

    Args:
        - self (instance): Instance argument required for plugging into Stats class.
        - dset (xr.Dataset): Dataset with variables to calculate distributions for.
        - var1 (str): Name of first variable in dset to compute joint distribution from.
        - var2 (str): Name of second variable in dset to compute joint distribution from.
        - var3 (str): Name of third variable in dset to compute joint distribution from.
        - bins1 (1d array): Bin edges for Hs.
        - bins2 (1d array): Bin edges for Tp.
        - bins3 (1d array): Bin edges for Dp.
        - dim (str): Dimension to calculate distribution along.
        - group (str): Time grouping type, any valid time_{group} such as month, season.

    Returns:
        - dsout (xr.Dataset): Dataset with Hs, Tp, Dp joint distributio along dim.

    Note:
        - The bins args can be a list/array with lower bin edges or a dictionary with
          np.arange kwargs 'start', 'stop' and 'step' ('start' and 'stop' are estimated
          from the data range if not available as keys).
        - Mask is defined based on missing values from first variable (typically hs).
        - Any directional variable should be provided as `var3` to ensure wrapping.

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
            "units": "m",
        },
        bin2_name: {
            "standard_name": f"{da2.attrs.get('standard_name', 'tp')}_bin",
            "long_name": f"{da2.attrs.get('long_name', 'tp')} bin",
            "units": "s",
        },
        bin3_name: {
            "standard_name": f"{da3.attrs.get('standard_name', 'dp')}_bin",
            "long_name": f"{da3.attrs.get('long_name', 'dp')} bin",
            "units": "degree",
        },
    }

    # Mask based on the first variable
    mask = da1.isel(**{dim: 0}, drop=True).notnull()

    # Grouping before computing
    if group:
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
    dsout[label].encoding = {"dtype": "int32", "_FillValue": -32768}

    return dsout


def distribution_spddir(
    spd,
    dir,
    spd_bins,
    dir_bins,
    dim="time",
    group="month",
    label="spd_dir_dist",
):
    """Joint wind distribution (spd, dir).

    Args:
        spd (xr.DataArray): Wind / current speed.
        dir (xr.DataArray): Wind / current direction.
        spd_bins (1d array): Bin edges for spd.
        dir_bins (1d array): Bin edges for dir.
        dim (str): Dimension to calculate distribution along.
        group (str): Time grouping type, any valid time_{group} such as month, season.
        label (str): Name for joint distribution variable.

    Returns:
        Dataset with Wspd, Wdir joint distributio along dim.

    """
    spd_bins = np.array(spd_bins)
    dir_bins = np.array(dir_bins)

    # Direction wrapping
    dir_bins = dir_bins - ((dir_bins[1] - dir_bins[0]) / 2)
    dir = _wrap_directions(dir, dirmax=dir_bins.max())

    # Bin coordinates at cell centre
    coords = {
        "spd_bin": spd_bins[:-1] + (spd_bins[1] - spd_bins[0]) / 2,
        "dir_bin": dir_bins[:-1] + (dir_bins[1] - dir_bins[0]) / 2,
    }

    # Mask based on spd
    mask = spd.isel(**{dim: 0}, drop=True).notnull()

    # Coordinates attributes
    attrs = {
        "spd_bin": {
            "standard_name": f"{spd.attrs.get('standard_name', 'speed')}_bin",
            "long_name": f"{spd.attrs.get('long_name', 'speed')} bin",
            "units": "m s-1",
        },
        "dir_bin": {
            "standard_name": f"{dir.attrs.get('standard_name', 'direction')}_bin",
            "long_name": f"{dir.attrs.get('long_name', 'direction')} bin",
            "units": "degree",
        },
    }

    # Grouping before computing
    if group:
        logger.debug(f"Grouping by {group}")
        spd = spd.groupby(f"time.{group}")
        dir = dir.groupby(f"time.{group}")

    # Computing
    dsout = (
        xr.apply_ufunc(
            _np_histogram_2d,
            spd,
            dir,
            spd_bins,
            dir_bins,
            input_core_dims=[[dim], [dim], ["dummy1"], ["dummy2"]],
            output_core_dims=[["spd_bin", "dir_bin"]],
            exclude_dims=set((dim,)),
            vectorize=True,
            dask="parallelized",
            output_dtypes=["int32"],
            dask_gufunc_kwargs={
                "output_sizes": {
                    "spd_bin": spd_bins.size - 1,
                    "dir_bin": dir_bins.size - 1,
                },
            },
        )
        .assign_coords(coords)
        .where(mask)
        .to_dataset(name=label)
    )

    # Attributes
    dsout[label].attrs = {
        "standard_name": "data_count",
        "long_name": "number of valid data points",
        "units": "",
    }
    dsout.spd_bin.attrs = attrs["spd_bin"]
    dsout.dir_bin.attrs = attrs["dir_bin"]
    dsout[label].encoding = {"dtype": "int32", "_FillValue": -32767}

    return dsout


def _wrap_directions(darr, dirmax):
    """Wrap directions DataArray."""
    return xr.where(darr <= dirmax, darr, darr - 360)


def _np_histogram_2d(arr1, arr2, bins1, bins2):
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


def _np_histogram_3d(arr1, arr2, arr3, bins1, bins2, bins3):
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


if __name__ == "__main__":
    dset = xr.open_dataset(
        "/source/onhindcast/implementation/swan/tasman/model/tasman-19790201T00-grid.nc",
        chunks={},
    )
    dsout = distribution(
        None,
        dset,
        var1="hs",
        var2="tps",
        var3="dpm",
        bins1={"step": 1.0},
        bins2=[0, 5, 10, 15, 20],
        bins3={"step": 90.0},
        dim="time",
        group="month",
    )