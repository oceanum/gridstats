"""Derived statistics from xarray."""
import numpy as np
import dask.array as da
import pandas as pd
from scipy.optimize import curve_fit

from onstats.utils import angle, wavelength, direc

DOUGLAS_SEA_INTERVALS = {
    0: pd.Interval(-np.inf, 0.0),
    1: pd.Interval(0.0, 0.1),
    2: pd.Interval(0.1, 0.5),
    3: pd.Interval(0.5, 1.25),
    4: pd.Interval(1.25, 2.5),
    5: pd.Interval(2.5, 4.0),
    6: pd.Interval(4.0, 6.0),
    7: pd.Interval(6.0, 9.0),
    8: pd.Interval(9.0, 14.0),
    9: pd.Interval(14.0, np.inf),
}

DOUGLAS_SWELL_INTERVALS = {
    0: {"height": pd.Interval(-np.inf, 0.0), "length": pd.Interval(-np.inf, 0.0)},
    1: {"height": pd.Interval(0.0, 2.0), "length": pd.Interval(0.0, 200.0)},
    2: {"height": pd.Interval(0.0, 2.0), "length": pd.Interval(200.0, np.inf)},
    3: {"height": pd.Interval(2.0, 4.0), "length": pd.Interval(0.0, 100.0)},
    4: {"height": pd.Interval(2.0, 4.0), "length": pd.Interval(100.0, 200.0)},
    5: {"height": pd.Interval(2.0, 4.0), "length": pd.Interval(200.0, np.inf)},
    6: {"height": pd.Interval(4.0, np.inf), "length": pd.Interval(0.0, 100.0)},
    7: {"height": pd.Interval(4.0, np.inf), "length": pd.Interval(100.0, 200.0)},
    8: {"height": pd.Interval(4.0, np.inf), "length": pd.Interval(200.0, np.inf)},
    9: {"height": pd.Interval(np.inf, np.inf), "length": pd.Interval(np.inf, np.inf)},
}


def douglas_sea(hs_sea, **kwargs):
    """Douglas sea scale.

    Args:
        hs_sea (xr.DataArray): Wind-sea significant wave height.

    """
    dsout = hs_sea.copy()
    for scale, interval in DOUGLAS_SEA_INTERVALS.items():
        dsout = dsout.where(
            cond=~((hs_sea > interval.left) & (hs_sea <= interval.right)), other=scale
        )
    dsout.attrs = {
        "long_name": "douglas sea scale",
        "standard_name": "sea_surface_wave_douglas_sea_scale",
        "units": "",
    }
    return dsout


def douglas_swell(hs_sw1, lp_sw1, **kwargs):
    """Douglas swell scale.

    Args:
        hs_sw1 (xr.DataArray): Primary swell significant wave height.
        hs_lp1 (xr.DataArray): Primary swell peak wave length.

    """
    dsout = hs_sw1.copy(deep=True)
    for scale, intervals in DOUGLAS_SWELL_INTERVALS.items():
        iheight = intervals["height"]
        ilength = intervals["length"]
        dsout = dsout.where(
            cond=~(
                (hs_sw1 > iheight.left)
                & (hs_sw1 <= iheight.right)
                & (lp_sw1 > ilength.left)
                & (lp_sw1 <= ilength.right)
            ),
            other=scale,
        )
    dsout.attrs = {
        "long_name": "douglas swell scale",
        "standard_name": "sea_surface_wave_douglas_swell_scale",
        "units": "",
    }
    return dsout


def wlen(fp=None, tp=None, depth=None):
    """Wevelength for the full wave spectrum.

    Args:
        fp (xr.DataArray): Total peak wave frequency (either `fp` or `tp` must be provided).
        tp (xr.DataArray): Total peak wave period (either `fp` or `tp` must be provided).
        depth (xr.DataArray): Water depth, if not provided the deep water approximation
            is returned.

    """
    dsout = wavelength(freq=fp, period=tp, depth=depth)
    dsout.attrs = {
        "long_name": "Wavelength of combined sea and swell",
        "standard_name": "sea_surface_wave_wavelength",
        "units": "m",
    }
    return dsout


def wlen_sea(fp=None, tp=None, depth=None):
    """Wevelength of the wind sea.

    Args:
        fp (xr.DataArray): Wind sea wave frequency (either `fp` or `tp` must be provided).
        tp (xr.DataArray): Wind sea wave period (either `fp` or `tp` must be provided).
        depth (xr.DataArray): Water depth, if not provided the deep water approximation
            is returned.

    """
    dsout = wavelength(freq=fp, period=tp, depth=depth)
    dsout.attrs = {
        "long_name": "Wavelength of wind sea",
        "standard_name": "sea_surface_wind_wave_wavelength",
        "units": "m",
    }
    return dsout


def wlen_sw1(fp=None, tp=None, depth=None):
    """Wevelength of the primary swell.

    Args:
        fp (xr.DataArray): Primary swell wave frequency (either `fp` or `tp` must be provided).
        tp (xr.DataArray): Primary swell wave period (either `fp` or `tp` must be provided).
        depth (xr.DataArray): Water depth, if not provided the deep water approximation
            is returned.

    """
    dsout = wavelength(freq=fp, period=tp, depth=depth)
    dsout.attrs = {
        "long_name": "Wavelength of primary swell",
        "standard_name": "sea_surface_primary_swell_wave_wavelength",
        "units": "m",
    }
    return dsout


def wspd(uwnd, vwnd, **kwargs):
    """Wind speed.

    Args:
        uwnd (DataArray): East-west component of wind velocity.
        vwnd (DataArray): North-south component of wind velocity.

    """
    dsout = da.sqrt(da.square(uwnd) + da.square(vwnd))
    dsout.attrs = {
        "long_name": "wind speed",
        "standard_name": "wind_speed",
        "units": "m/s",
    }
    return dsout


def wdir(uwnd, vwnd, **kwargs):
    """Wind coming-from direction.

    Args:
        uwnd (DataArray): East-west component of wind velocity.
        vwnd (DataArray): North-south component of wind velocity.

    """
    dsout = direc(uwnd, vwnd)
    dsout.attrs = {
        "long_name": "wind from direction",
        "standard_name": "wind_from_direction",
        "units": "degree",
    }
    return dsout


def clear_sky(cloud_cover, cover_threshold=0.0, **kwargs):
    """Wind speed.

    Args:
        cloud_cover (DataArray): Cloud area fraction.
        cover_threshold (float): Maximum cover fraction for assuming clear sky.

    """
    dsout = cloud_cover <= cover_threshold
    dsout.attrs = {"long_name": "clear sky", "standard_name": "clear_sky", "units": ""}
    return dsout


def covered_sky(cloud_cover, cover_threshold=1.0, **kwargs):
    """Wind speed.

    Args:
        cloud_cover (DataArray): Cloud area fraction.
        cover_threshold (float): Minimum cover fraction for assuming covered sky.

    """
    dsout = cloud_cover >= cover_threshold
    dsout.attrs = {
        "long_name": "covered sky",
        "standard_name": "covered_sky",
        "units": "",
    }
    return dsout


def cspd(ucur, vcur, **kwargs):
    """urrent speed.

    Args:
        ucur (DataArray): East-west component of current velocity.
        vcur (DataArray): North-south component of current velocity.

    """
    dsout = da.sqrt(da.square(ucur) + da.square(vcur))
    dsout.attrs = {
        "long_name": "sea water speed",
        "standard_name": "sea_water_speed",
        "units": "m/s",
    }
    return dsout


def crossing_seas(
    hs,
    hs_sea,
    hs_sw1,
    dir_sea,
    dir_sw1,
    hs_sw2=None,
    dir_sw2=None,
    hs_threshold=0,
    **kwards
):
    """Crossing seas occurrance.

    Args:
        hs (DataArray): Total significant wave height.
        hs_sea (xr.DataArray): Wind-sea significant wave height.
        hs_sw1 (xr.DataArray): Primary swell significant wave height.
        hs_sw2 (xr.DataArray): Secondary swell significant wave height.
        dir_sea (xr.DataArray): Wind-sea wave direction.
        dir_sw1 (xr.DataArray): Primary swell wave direction.
        dir_sw2 (xr.DataArray): Secondary swell wave direction.
        hs_threshold (float): Minimum Hs above which a partition considered.

    Returns:
        Bool DataArray specifying where crossing seas are identified.

    Crossing seas are identified according to two criteria:
        (1) Relative angle between wave systems is larger than 40 degrees.
        (2) Less energetic system has at least 20% of the total enerty.
    A third criteria based on minimum total Hs is also applied using the hs_threshold.

    Note:
        If secondary partitions are not provided crossing seas are only defined for the
            wind sea and primary swell partitions.

    Reference:
        Li, X.M. (2016). A new insight from space into swell propagation and
            crossing in the global oceans, Geophysical Research Letters,
            43(10), P. 5202-5209.

    """
    hs_gt_threshold = hs.where(hs > hs_threshold)

    hs_sea_gt_20 = hs_sea > np.sqrt(0.2) * hs_gt_threshold
    hs_sw1_gt_20 = hs_sw1 > np.sqrt(0.2) * hs_gt_threshold
    hs_sw2_gt_20 = hs_sw2 > np.sqrt(0.2) * hs_gt_threshold

    c1 = da.logical_and(
        da.logical_and(hs_sea_gt_20, hs_sw1_gt_20), angle(dir_sea, dir_sw1) > 40
    )
    c2 = da.logical_and(
        da.logical_and(hs_sea_gt_20, hs_sw2_gt_20), angle(dir_sea, dir_sw2) > 40
    )
    c3 = da.logical_and(
        da.logical_and(hs_sw1_gt_20, hs_sw2_gt_20), angle(dir_sw1, dir_sw2) > 40
    )

    dsout = da.logical_or(da.logical_or(c1, c2), c3)

    dsout.attrs = {
        "long_name": "crossing seas",
        "standard_name": "sea_surface_crossing_seas",
        "units": "",
        "hs_threshold": hs_threshold,
    }
    return dsout


def winpow(uwnd150, vwnd150):
    """Wind Power for Wind Quarry turbine.

    Args:
        u150 (array): Eastward wind component at 150 m elevation.
        v150 (array): Northward wind component at 150 m elevation.

    """
    def func(x, a, b, c, d):
        return (a * x) + (b * x**2) + (c * x**3) + d

    x = np.array([3.0, 3.6, 4.32, 5.06, 6.2, 7.05, 8.83, 10.61])
    y = np.array([1500, 1915, 2719, 3872, 6110, 7963, 13490, 16000])
    fits, __ = curve_fit(func, x, y)

    wspd150 = wspd(uwnd150, vwnd150)
    power = func(wspd150, *fits)

    # Zero below cut-in
    power = power.where(wspd150 > 3, 0)
    # Maximum between rated and cut-out
    power = power.where(wspd150 < 10.61, 16000)
    # Zero above cut-out
    power = power.where(wspd150 < 25, 0)

    return power
