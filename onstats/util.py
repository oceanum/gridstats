"""Auxiliary functions."""
import numpy as np
import dask.array as da
import pandas as pd
import xarray as xr
from scipy.signal import find_peaks
from subprocess import run, PIPE, STDOUT


def pot(data, perc=95, duration=24):
    """Peaks over threshold.

    Args:
        data (pd.Series): Timeseries data to select peaks from.
        perc (float): Percentile above which peaks are selected.
        duration (float): Hours in storm below which extra peaks are discarded.

    Return:
        Subset pandas series with data for selected peaks.

    """
    # Only supporting 1d array for now
    if not isinstance(data, pd.Series):
        raise ValueError("Only pandas Series are supported")

    # Ensure no missing values in timeseries
    if any(data.isna()):
        raise ValueError("Peak over threshold does not support missing values")

    # Ensure regularly-spaced imeseries
    tdiff = np.diff(data.index)
    if tdiff.min() != tdiff.max():
        raise ValueError("Peaks over threshold requires regular timesteps in data")

    dt = pd.to_timedelta(tdiff[0]).total_seconds() / 3600
    distance = duration / dt
    ind_perc = int(0.01 * perc * len(data))
    height = data.sort_values()[ind_perc]
    ind = find_peaks(data, height=height, distance=distance)[0]
    return data.iloc[ind]


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
    ang_rot = 180 if coming_from else 0
    vetor = u + v * 1j
    mag = da.absolute(vetor)
    direc = xr.ufuncs.angle(vetor, deg=True) + ang_rot
    direc = da.mod(90 - direc, 360)
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


if __name__ == "__main__":

    """
    # With timezoneFinder
    tf = TimezoneFinder()
    tf = TimezoneFinder(in_memory=True)
    longitude = 13.358
    latitude = 52.5061
    tf.timezone_at(lng=longitude, lat=latitude)  # returns 'Europe/Berlin'
    import datetime
    import pytz

    # With tzwhere
    tzwhere = tzwhere.tzwhere()
    timezone_str = tzwhere.tzNameAt(37.3880961, -5.9823299)  # Seville coordinates
    timezone_str
    # > Europe/Madrid

    timezone = pytz.timezone(timezone_str)
    dt = datetime.datetime.now()
    timezone.utcoffset(dt)
    # > datetime.timedelta(0, 7200)

    01 - 0
    14 - 0
    15 - 1
    20 - 1
    29 - 1
    30 - 2

    """
    import pytz
    from timezonefinder import TimezoneFinder
    from tzwhere import tzwhere

    # Trying out
    dset = xr.open_dataset(
        "/data/swan/cfsr_st6-01/swn20160101_00z/nwwa_20160101T00_grid.nc"
    )
    dset.attrs = {}
    ds = dset.isel(latitude=slice(None, None, 5), longitude=slice(None, None, 5)).load()

    # Make time 3D
    ds["time3d"] = ds.time.expand_dims(
        dim={"latitude": ds.latitude.size, "longitude": ds.longitude.size}, axis=(1, 2)
    )

    lat2, lon2 = xr.broadcast(ds.latitude, ds.longitude)
    lat1 = lat2.stack(dimension=("latitude", "longitude"))
    lon1 = lon2.stack(dimension=("latitude", "longitude"))

    # t = timeday.isel(time=slice(0, 6), latitude=slice(0, 4), longitude=slice(0, 2))
    hour_offset = np.floor((ds.longitude + 7.7) / 15)
