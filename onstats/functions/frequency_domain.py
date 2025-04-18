"""Frequency domain processing."""
import numpy as np
import dask.array as da
import xarray as xr
from scipy.signal import welch
from scipy.interpolate import interp1d
from scipy.integrate import simpson


BANDS = {
    "0_25": (1 / 25, None),
    "8_25": (1 / 25, 1 / 8),
    "25_120": (1 / 120, 1 / 25),
    "25_300": (1 / 300, 1 / 25),
    "tot": (None, None),
}


def _np_hs(x, fs, segsec, bands):
    """Frequency domain significant wave height for frequency bands.

    Args:
        x (1darr): Time series of measurement values.
        fs (float): Sampling frequency of x (Hz).
        segsec (int): Size of overlapping segments (s).
        bands (2darr): Frequency bands with [fmin, fmax] for each band in a row (Hz).

    Note:
        Default hann window is used with default overlapping of 50%.
        Default "constant" detrend is used (subtract mean of each segment).
        x must be regularly-spaced and with no gaps.
        Length of fft is defined by the next power of 2 from nperseg.

    """
    # Calculate segment size
    nperseg = segsec * fs
    nfft = int(2 ** np.ceil(np.log2(nperseg + 1)))

    # Calculate power spectrum
    freq, efth = welch(x, fs=fs, nperseg=nperseg, nfft=nfft)
    f = interp1d(freq, efth)

    # Calculate Hs for each band
    fmins = np.nan_to_num(bands[:, 0], nan=freq[0])
    fmaxs = np.nan_to_num(bands[:, 1], nan=freq[-1])
    hs = []
    for fmin, fmax in zip(fmins, fmaxs):
        ifreq = (freq > fmin) & (freq < fmax)
        freq_band = np.hstack((fmin, freq[ifreq], fmax))
        efth_band = f(freq_band)
        hs.append(float(4 * np.sqrt(simpson(efth_band, freq_band))))
    return np.array(hs)


def hmo(self, dset, segsec=256, bands=BANDS, dim="time", group=None):
    """Frequency domain significant wave height for frequency bands.

    Args:
        - self (instance): Instance argument required for plugging into Stats class.
        - dset (xr.Dataset): Dataset with variables to calculate Hmo for.
        - segsec (int): Size of overlapping segments (s).
        - bands (dict): Frequency bands, keys are band labels, values are [fmin, fmax] (Hz).
        - dim (str): Dimension to calculate fft along.

    Returns:
        - hs (DataArray):

    Note:
        - None or nan band values are interpreted as the min or max frequency available.

    """
    # Apply function
    bands_array = np.array(list(bands.values()), dtype=float)
    fs = 1 / float(np.diff(dset[dim]).mean())
    dsout = xr.apply_ufunc(
        _np_hs,
        dset,
        fs,
        segsec,
        bands_array,
        input_core_dims=[[dim], [], [], ["band", "fbounds"]],
        output_core_dims=[["band"]],
        exclude_dims=set((dim,)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    dsout = dsout.assign_coords({"band": list(bands.keys())})
    # Finalise
    dsout.band.attrs = {
        "standard_name": "frequency_band",
        "long_name": "Frequency ranges hs is calculated over",
        "units": "Hz",
    }
    if isinstance(dset, xr.DataArray):
        dsout = dsout.to_dataset()
    for varname, dvar in dsout.data_vars.items():
        dvar.attrs = {
            "standard_name": "sea_surface_wave_significant_height",
            "long_name": "frequency-domain significant wave height of frequency bands",
            "units": dvar.attrs.get("units", "m"),
        }
    dsout = dsout.rename({v: f"hs_{v}" for v in dsout.data_vars})
    if isinstance(dset, xr.DataArray):
        dsout = dsout.to_array().isel(variable=0, drop=True)
        dsout.name = f"hs_{dset.name}"
    return dsout
