"""Probability stats."""
import numpy as np
import dask.array as da
import xarray as xr
import logging


logger = logging.getLogger(__name__)


def range_probability(self, dset, data_ranges, dim="time", **kwargs):
    """Calculate probability of specific ranges.

    Args:
        - self (instance): Instance argument required for plugging into Stats class.
        - dset (xr.Dataset): Dataset with variables to calculate probabilities for.
        - data_ranges (list): List of dictionaries defining data ranges to calculate
          probabilities over with keys:
            - var (str): Variable name.
            - start (float, None): Minimum value for interval.
            - stop (float, None); Maximum value for interval.
            - left (closed | open): Define if minimum value should be included.
            - right (closed | open): Define if maximum value should be included.
            - label (str): Name for this variable in output dataset.
        - dim (str): Dimension to calculate range probability along.

    """
    if not isinstance(data_ranges, list):
        data_ranges = [data_ranges]

    logger.warning(f"kwargs ignored: {kwargs}")

    left_funcs = {"closed": da.greater_equal, "open": da.greater}
    right_funcs = {"closed": da.less_equal, "open": da.less}

    dsout = xr.Dataset()
    for data_range in data_ranges:

        # Data variable to compute
        dvar = data_range["var"]
        darr = dset[dvar]

        # Data range values
        start = data_range["start"] if data_range["start"] is not None else -np.inf
        stop = data_range["stop"] if data_range["stop"] is not None else np.inf

        # Choose functions depending on interval is closed or open
        left = data_range.get("left", "closed")
        right = data_range.get("right", "closed")
        lfunc = left_funcs[left]
        rfunc = right_funcs[right]

        # Output label
        llabel = f"{start:g}" if data_range["start"] is not None else "min"
        rlabel = f"{stop:g}" if data_range["stop"] is not None else "max"
        varname = data_range.get("label", f"{dvar}_{llabel}_to_{rlabel}")

        # Probability
        in_range = lfunc(darr, start) & rfunc(darr, stop)
        dsout[varname] = in_range.sum(dim=dim) / darr.count(dim)

    return dsout
