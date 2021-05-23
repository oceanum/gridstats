"""Frequency domain processing."""
import numpy as np
import xarray as xr
from scipy.signal import welch
from scipy.interpolate import interp1d
from scipy.integrate import simps


BANDS = {
    "0_25": (1 / 25, 0),
    "8_25": (1 / 25, 1 / 8),
    "25_120": (1 / 120, 1 / 25),
    "25_300": (1 / 300, 1 / 25),
    "0_300": (1 / 300, 0),
}


def np_hs(x, fs, segsec, bands):
    """Frequency domain significant wave height for frequency bands.

    Args:
        x (1darr): Time series of measurement values
        fs (float): Sampling frequency of x
        segsec (int): Size of overlapping segments (s).
        bands (list): Frequency bands with [fmin, fmax] for each band to calculate.

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
    f0 = freq[0]
    f1 = freq[-1]
    hs = []
    for band in bands:
        fmin = band[0] or f0
        fmax = band[1] or f1
        ifreq = (freq > fmin) & (freq < fmax)
        freq_band = np.hstack((fmin, freq[ifreq], fmax))
        efth_band = f(freq_band)
        hs.append(4 * np.sqrt(simps(efth_band, freq_band)))
    return np.array(hs)


def hmo(darr, fs, segsec=256, bands=BANDS, dim="second"):
    """Frequency domain significant wave height for frequency bands.

    Args:
        darr (DataArray): Time series data to calculate Hs from.
        fs (float): Sampling frequency of x
        segsec (int): Size of overlapping segments (s).
        bands (dict): Frequency bands, keys are band labels, values are [fmin, fmax].
        dim (str): Dimension to calculate fft along.

    Returns:
        hs (DataArray): 

    """
    # Apply function
    bands_list = np.array(list(bands.values()))
    dsout = xr.apply_ufunc(
        np_hs,
        darr,
        fs,
        segsec,
        bands_list,
        input_core_dims=[[dim], [], [], ["band", "fbounds"]],
        output_core_dims=[["band"]],
        exclude_dims=set((dim,)),
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"],
    )
    dsout = dsout.assign_coords({"band": list(bands.keys())})
    # Finalise
    dsout.band.attrs = {
        "standard_name": "frequency_band",
        "long_name": "Frequency ranges hs is calculated over",
        "units": "Hz",
    }
    if isinstance(darr, xr.DataArray):
        dsout = dsout.to_dataset()
    for varname, dvar in dsout.data_vars.items():
        dvar.attrs = {
            "standard_name": "sea_surface_wave_significant_height",
            "long_name": "frequency-domain significant wave height of frequency bands",
            "units": darr[varname].attrs.get("units", "m"),
        }
    dsout = dsout.rename({v: f"hs_{v}" for v in dsout.data_vars})
    if isinstance(darr, xr.DataArray):
        dsout = dsout.to_array().isel(variable=0, drop=True)
        dsout.name = f"hs_{darr.name}"
    return dsout
