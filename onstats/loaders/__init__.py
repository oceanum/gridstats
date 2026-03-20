"""Data loading plugins for onstats.

Built-in loaders (xarray, intake) self-register on import.
"""
from onstats.loaders.xarray import XarrayLoader
from onstats.loaders.intake import IntakeLoader

__all__ = ["XarrayLoader", "IntakeLoader"]
