"""Auxiliary functions."""
import logging
import numpy as np
import dask.array as da
import pandas as pd
import xarray as xr
from scipy import stats, signal
from subprocess import run, PIPE, STDOUT
from inspect import signature, Parameter
from functools import wraps


logger = logging.getLogger(__name__)


def _get_default_parameters(func):
    """Default kwargs from function."""
    param = signature(func).parameters
    kw = {k: v.default for k, v in param.items() if v.default is not Parameter.empty}
    return kw


def stepwise(func):
    """Execute func on loaded slices of dataset in a stepswise manner.

    This decorator is intended to run functions requiring single time chunks (e.g. rpv,
        quantile) onto large datasets that are not chunked with single time chunks and
        require rechunking which can require prohibitively large amounts of RAM.

    Required signature in decorated function func:
        - self as the first arg as function is attached to Stats class dynamically.
        - dset as either a kwarg or the second arg.
        - yname, xname specifying dimension names to stepwise through.
        - ystep, xstep specifying dimension sizes to stepwise through.

    """
    @wraps(func)
    def wrapped_func(*args, **kwargs):

        # Stepwise applied only if both steps are provided
        if kwargs.get("ystep", None) is None and kwargs.get("xstep", None) is None:
            logger.info("No stepwise applied")
            return func(*args, **kwargs)

        # Default kwargs that may have not been set
        kwdef = _get_default_parameters(func)

        # Ensure function has required kwargs
        for key in ["yname", "xname", "ystep", "xstep"]:
            if key not in kwargs and key not in kwdef:
                raise ValueError(
                    f"stepwise decorator requires '{key}' kwarg in func but the "
                    f"func signature is: {func.__name__}{signature(func)}"
                )

        # Dataset
        dset = kwargs.pop("dset", args[1])
        if not isinstance(dset, xr.Dataset):
            ValueError("stepwise decorator requires dset as a kwarg or the first arg")

        # Coords names
        yname = kwargs.pop("yname", kwdef["yname"])
        xname = kwargs.pop("xname", kwdef["xname"])
        for name in [yname, xname]:
            if name not in dset.dims:
                raise ValueError(f"'{name}' is not a dimension of dataset {dset.dims}")

        # Dims and steps sizes
        yend = dset[yname].size
        xend = dset[xname].size
        ystep = kwargs.pop("ystep", kwdef["ystep"]) or yend
        xstep = kwargs.pop("xstep", kwdef["xstep"]) or xend
        if yend % ystep != 0:
            yend += ystep
        if xend % xstep != 0:
            xend += xstep

        # Steps intervals
        y_intervals = pd.interval_range(start=0, end=yend, freq=ystep)
        x_intervals = pd.interval_range(start=0, end=xend, freq=xstep)
        n_intervals = len(y_intervals) * len(x_intervals)

        # Compute each spatial box slice loading before calculating stats
        i = 1
        dsout_list = []
        for iy, yint in enumerate(y_intervals):
            for xint in x_intervals:
                logger.info(f"Compute partial dataset {i}/{n_intervals}")
                slice_kwargs = {
                    yname: slice(yint.left, yint.right),
                    xname: slice(xint.left, xint.right),
                }
                ds = dset.isel(slice_kwargs).load()
                # Function needs to have self and dset as 1st and 2nd args
                dsout = func(args[0], ds, *args[2:], **kwargs)
                dsout_list.append(dsout)
                i += 1
        dsout = xr.combine_by_coords(dsout_list)
        return dsout

    return wrapped_func


def expand_time_group(dset, time_group):
    """Concatenate time grouped dataset into new dimension.

    Args:
        dset (xr.Dataset): Dataset to groupby for.
        time_group (str): Name of time group to apply, e.g. "month", "season".

    """
    groups = dset.groupby(f"time.{time_group}")
    dsout = xr.concat([g[1] for g in groups], dim=time_group)
    dsout[time_group] = [g[0] for g in groups]
    dsout[time_group].attrs = {
        "standard_name": time_group,
        "long_name": time_group,
        "units": "",
    }
    return dsout


def run_command(cmd):
    process = run(cmd, shell=True, stdout=PIPE, stderr=STDOUT)
    process.check_returncode()
    return process


def grid_smooth(darr, n=10):
    """Smooth Data Array.

    Args:
        darr (DataArray): Data array to smooth.
        n (int): number of iterations - the larger, the smoother, choose between 10-30.

    """

    # left = xr.concat((darr, darr.isel(longitude=-1)), dim='longitude').isel(longitude=slice(1, None))
    # right = xr.concat((darr.isel(longitude=0), darr), dim='longitude').isel(longitude=slice(None, -1))
    # bottom = xr.concat((darr.isel(latitude=0), darr), dim='latitude').isel(latitude=slice(None, -1))
    # top = xr.concat((darr, darr.isel(latitude=-1), darr), dim='latitude').isel(latitude=slice(1, None))
    # darr = 0.125 * (left + right + top + bottom) + 0.5 * darr
    # darr = np.ma.array(darr)
    # if n > 1:
    #     return grid_smooth(darr, n-1)
    # else:
    #     return darr
    dout = darr.copy(deep=True)
    val = dout.fillna(0.0).values
    left = np.hstack((val, val[:, -1:]))
    right = np.hstack((val[:, :1], val))
    top = np.vstack((val, val[-1:, :]))
    bot = np.vstack((val[0], val))
    val = 0.125 * (left[:, 1:] + right[:, :-1] + top[1:, :] + bot[:-1, :]) + 0.5 * val
    dout.values = val
    if n > 1:
        return grid_smooth(dout, n - 1)
    else:
        return dout


def unique_times(dset):
    """Remove duplicate times from dataset."""
    __, index = da.unique(dset["time"], return_index=True)
    return dset.isel(time=index)


def to_datetime(np64):
    """Convert Datetime64 date to datatime."""
    if isinstance(np64, np.datetime64):
        dt = pd.to_datetime(str(np64)).to_pydatetime()
    elif isinstance(np64, xr.DataArray):
        dt = pd.to_datetime(str(np64.values)).to_pydatetime()
    else:
        IOError(
            "Cannot convert {} into datetime, expected np.datetime64 or xr.DataArray".format(
                type(np64)
            )
        )
    return dt


def spddir_to_uv(spd, direc, coming_from=False):
    """Converts (spd, dir) to (u, v).

    Args:
        spd (array): Magnitudes to convert.
        direc (array): Directions to convert (degree).
        coming_from (bool): True if directions in coming-from convention, False if in going-to.

    Returns:
        u (array): Eastward wind component.
        v (array): Northward wind component.

    """
    ang_rot = 180 if coming_from else 0
    direc_rotate = da.deg2rad(direc + ang_rot)
    u = spd * da.sin(direc_rotate)
    v = spd * da.cos(direc_rotate)
    return u, v


def uv_to_spddir(u, v, coming_from=False):
    """Converts (u, v) to (spd, dir).

    Args:
        u (array): Eastward wind component.
        v (array): Northward wind component.
        coming_from (bool): True for output directions in coming-from convention, False for going-to.

    Returns:
        mag (array): magnitudes.
        direc (array): directions (degree).

    """
    to_nautical = 270 if coming_from else 90
    mag = da.sqrt(u ** 2 + v ** 2)
    direc = da.rad2deg(da.arctan2(v, u))
    direc = (to_nautical - direc) % 360
    return mag, direc


def direc(u, v, coming_from=False):
    """Calculated direction from (u, v) components.

    Args:
        u (array): Eastward wind component.
        v (array): Northward wind component.
        coming_from (bool): True for output directions in coming-from convention, False for going-to.

    Returns:
        direc (array): directions (degree).

    """
    ang_rot = 180 if coming_from else 0
    vetor = u + v * 1j
    direc = xr.ufuncs.angle(vetor, deg=True) + ang_rot
    return da.mod(90 - direc, 360)


def angle(dir1, dir2, **kwargs):
    """Relative angle between two directions.

    Args:
        dir1 (DataArray): First angles to compare.
        dir1 (DataArray): Second angles to compare.

    """
    dif = da.absolute(dir1 % 360 - dir2 % 360)
    return da.minimum(dif, 360 - dif)


def wavelength(freq=None, period=None, depth=None):
    """Wavelength L.

    Args:
        freq (ndarray): Wave frequency (either `freq` or `period` must be provided).
        period (ndarray): Wave period (either `freq` or `period` must be provided).
        depth (ndarray): Water depth, if not provided the deep water approximation
            is returned.

    Returns;
        wavelength ndarray.

    """
    assert (
        freq is not None or period is not None
    ), "Either freq or period must be provided"
    if freq is None:
        freq = 1.0 / period
    if depth is not None:
        ang_freq = 2 * np.pi * freq
        return 2 * np.pi / wavenumber(ang_freq, depth)
    else:
        return 1.56 / freq ** 2


def wavenumber(ang_freq, water_depth):
    """Chen and Thomson wavenumber approximation.

    Args:
        ang_freq (array): Angular frequency 2*pi*f.
        water_depth (array): Water depth.

    """
    k0h = 0.10194 * ang_freq * ang_freq * water_depth
    D = [0, 0.6522, 0.4622, 0, 0.0864, 0.0675]
    a = 1.0
    for i in range(1, 6):
        a += D[i] * k0h ** i
    return (k0h * (1 + 1.0 / (k0h * a)) ** 0.5) / water_depth
