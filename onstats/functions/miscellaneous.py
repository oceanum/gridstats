"""Miscellaneous functions."""


def pcount(self, dset, dim, **kwargs):
    """Percentaga count function.

    Args:
        dset (xr.Dataset): Dataset to reduce.
        dim (str): Dimension along which percentage count is calculated.

    Returns:
        dsout (xr.Dataset): Reduced dataset.

    """
    dsout = 100 * dset.count(dim) / dset[dim].size
    return dsout
