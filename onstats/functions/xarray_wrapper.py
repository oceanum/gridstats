"""Wrap xarray stats methods to have the required Plugin signature."""
import xarray as xr


def min(self, dset, **kwargs):
    return dset.min(**kwargs)


def max(self, dset, **kwargs):
    return dset.max(**kwargs)


def mean(self, dset, **kwargs):
    return dset.mean(**kwargs)


def std(self, dset, **kwargs):
    return dset.std(**kwargs)


def quantile(self, dset, **kwargs):
    return dset.quantile(**kwargs)