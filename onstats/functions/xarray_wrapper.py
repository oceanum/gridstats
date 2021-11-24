"""Wrap xarray stats methods to have the required Plugin signature."""
import logging
import xarray as xr


logger = logging.getLogger(__name__)


def _groupby(dset, group):
    """Groupby dataset.

    Args:
        dset (xr.Dataset): Dataset to groupby.
        group (str): Time grouping type, any valid time_{group} such month, season.

    Returns:
        dset (Dataset, DatasetGroupBy): Dataset grouped by or not.

    """
    if group is not None:
        logger.info(f"Grouping by {group}")
        dset = dset.groupby(f"time.{group}")
    return dset


def min(self, dset, group=None, **kwargs):
    """Minimum wrapper function.

    Args:
        dset (xr.Dataset): Dataset to reduce.
        group (str): Time grouping type, any valid time_{group} such month, season.
        kwargs: Keywork arguments to pass to xarray's min method.

    Returns:
        dsout (xr.Dataset): Reduced dataset.

    """
    dset = _groupby(dset, group)
    return dset.min(**kwargs)


def max(self, dset, group=None, **kwargs):
    """Maximum wrapper function.

    Args:
        dset (xr.Dataset): Dataset to reduce.
        group (str): Time grouping type, any valid time_{group} such month, season.
        kwargs: Keywork arguments to pass to xarray's max method.

    Returns:
        dsout (xr.Dataset): Reduced dataset.

    """
    dset = _groupby(dset, group)
    return dset.max(**kwargs)


def mean(self, dset, group=None, **kwargs):
    """Mean wrapper function.

    Args:
        dset (xr.Dataset): Dataset to reduce.
        group (str): Time grouping type, any valid time_{group} such month, season.
        kwargs: Keywork arguments to pass to xarray's mean method.

    Returns:
        dsout (xr.Dataset): Reduced dataset.

    """
    dset = _groupby(dset, group)
    return dset.mean(**kwargs)


def std(self, dset, group=None, **kwargs):
    """Standard deviation wrapper function.

    Args:
        dset (xr.Dataset): Dataset to reduce.
        group (str): Time grouping type, any valid time_{group} such month, season.
        kwargs: Keywork arguments to pass to xarray's std method.

    Returns:
        dsout (xr.Dataset): Reduced dataset.

    """
    dset = _groupby(dset, group)
    return dset.std(**kwargs)


def quantile(self, dset, group=None, **kwargs):
    """Quantile wrapper function.

    Args:
        dset (xr.Dataset): Dataset to reduce.
        group (str): Time grouping type, any valid time_{group} such month, season.
        kwargs: Keywork arguments to pass to xarray's quantile method.

    Returns:
        dsout (xr.Dataset): Reduced dataset.

    """
    dset = _groupby(dset, group)
    return dset.quantile(**kwargs)
