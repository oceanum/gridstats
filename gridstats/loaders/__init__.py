"""Data loading plugins for gridstats.

Built-in loaders (xarray, intake) self-register on import.
"""
from gridstats.loaders.xarray import XarrayLoader
from gridstats.loaders.intake import IntakeLoader

__all__ = ["XarrayLoader", "IntakeLoader"]
