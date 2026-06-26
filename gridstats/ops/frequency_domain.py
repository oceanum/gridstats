"""Frequency-domain (spectral) statistics."""
from __future__ import annotations

import logging

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.signal import welch
import xarray as xr

from gridstats.registry import register_stat

logger = logging.getLogger(__name__)

# Default frequency bands in Hz: {label: [fmin, fmax]}  (None → use data limits)
BANDS: dict[str, list[float | None]] = {
    "0_25": [None, 0.25],
    "8_25": [0.08, 0.25],
    "25_120": [0.25, 1.20],
    "25_300": [0.25, 3.00],
    "tot": [None, None],
}


def _np_hs(x: np.ndarray, fs: float, segsec: float, bands: np.ndarray) -> np.ndarray:
    """Compute significant wave height in frequency bands via Welch PSD.

    Args:
        x: 1-D signal array.
        fs: Sampling frequency (Hz).
        segsec: Segment length in seconds for Welch method.
        bands: Array of shape (n_bands, 2) with [fmin, fmax] per band.

    Returns:
        Array of shape (n_bands,) with Hs values in metres.
    """
    nperseg = int(segsec * fs)
    freq, psd = welch(x, fs=fs, nperseg=nperseg)
    f = interp1d(freq, psd, bounds_error=False, fill_value=0.0)

    fmins = np.nan_to_num(bands[:, 0], nan=freq[0])
    fmaxs = np.nan_to_num(bands[:, 1], nan=freq[-1])
    hs = []
    for fmin, fmax in zip(fmins, fmaxs):
        ifreq = (freq > fmin) & (freq < fmax)
        freq_band = np.hstack((fmin, freq[ifreq], fmax))
        efth_band = f(freq_band)
        hs.append(float(4 * np.sqrt(simpson(efth_band, x=freq_band))))
    return np.array(hs, dtype="float32")


def _infer_fs(times: np.ndarray, ndigits: int = 3) -> float:
    """Infer the sampling frequency (Hz) from a time coordinate.

    Uses the mean sampling interval over the whole record rather than a single
    interval, which is robust to small timing jitter (e.g. model output whose
    nominal 1 Hz cadence wanders by a few percent because the internal time
    step varies). The mean interval is rounded to ``ndigits`` decimal places so
    that a nominal-1 s cadence collapses cleanly to ``dt = 1.0`` / ``fs = 1.0``.

    Args:
        times: 1-D array of time values. ``datetime64`` arrays are converted
            from nanoseconds to seconds; numeric arrays are assumed to be in
            seconds already.
        ndigits: Decimal places to round the mean interval (seconds) to.

    Returns:
        Sampling frequency in Hz.
    """
    span = times[-1] - times[0]
    if np.issubdtype(times.dtype, np.datetime64):
        span = span / np.timedelta64(1, "s")
    dt = round(float(span) / (times.size - 1), ndigits)
    return 1.0 / dt


@register_stat("hmo")
def hmo(
    data: xr.Dataset,
    *,
    dim: str = "time",
    segsec: float = 256,
    bands: dict[str, list[float | None]] = BANDS,
    fs: float | None = None,
    group: str | None = None,
    **kwargs,
) -> xr.Dataset:
    """Compute frequency-domain significant wave height (Hmo) per frequency band.

    Uses Welch's method to estimate the power spectral density, then integrates
    it over specified frequency bands: Hs = 4 * sqrt(integral(S(f) df)).

    Args:
        data: Input dataset. Variables must be time-series of surface elevation
            or equivalent.
        dim: Time dimension name.
        segsec: Welch segment length in seconds.
        bands: Mapping of band label to [fmin, fmax] in Hz.
            None values are replaced by the data's min/max frequency.
        fs: Sampling frequency in Hz. If ``None`` (default) it is inferred from
            the mean sampling interval of ``dim`` (rounded to 3 decimal places),
            which tolerates small timing jitter. Set explicitly to override.
        group: Time component to group by (not typically meaningful for Hmo).

    Returns:
        Dataset with variables named 'hs_{band_label}', each with a 'band'
        coordinate.
    """
    bands_array = np.array(list(bands.values()), dtype=float)
    if fs is None:
        fs = _infer_fs(data[dim].values)

    dsout = xr.apply_ufunc(
        _np_hs,
        data,
        fs,
        segsec,
        bands_array,
        input_core_dims=[[dim], [], [], ["band", "fbounds"]],
        output_core_dims=[["band"]],
        exclude_dims={dim},
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    dsout = dsout.assign_coords({"band": list(bands.keys())})
    dsout["band"].attrs = {
        "standard_name": "frequency_band",
        "long_name": "Frequency ranges Hs is calculated over",
        "units": "Hz",
    }
    for varname in dsout.data_vars:
        dsout[varname].attrs = {
            "standard_name": "sea_surface_wave_significant_height",
            "long_name": "frequency-domain significant wave height",
            "units": "m",
        }
    return dsout.rename({v: f"hs_{v}" for v in dsout.data_vars})
