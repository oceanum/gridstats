"""xarray-based data loader."""
from __future__ import annotations

import logging
from typing import Any

import xarray as xr

from gridstats.config import XarraySourceConfig, _BaseSourceConfig
from gridstats.registry import register_loader

logger = logging.getLogger(__name__)


def _ds_summary(dset: xr.Dataset) -> str:
    """Return a one-line summary of dataset shape and chunking for logging."""
    sizes = dict(dset.sizes)
    if dset.chunks:
        chunks = {d: sorted(set(c)) for d, c in dset.chunks.items()}
    else:
        chunks = "not chunked"
    return f"vars={list(dset.data_vars)}, sizes={sizes}, chunks={chunks}"


def _parse_sel_value(val: Any) -> Any:
    """Convert a ``{start, stop}`` dict to a :class:`slice`; pass everything else through.

    This allows YAML range specifications like ``{start: -50, stop: -30}`` to
    be used in ``sel`` without serialising Python ``slice`` objects. Either key
    may be omitted: ``{stop: 0}`` becomes ``slice(None, 0)``.

    Anything that is not a dict with at least one of ``start`` / ``stop`` is
    returned unchanged, so scalar values, lists, and other types keep their
    xarray-native semantics.
    """
    if isinstance(val, dict) and ("start" in val or "stop" in val):
        return slice(val.get("start"), val.get("stop"))
    return val


@register_loader("xarray")
class XarrayLoader:
    """Load datasets using xarray.open_dataset / open_zarr."""

    def load(self, config: XarraySourceConfig) -> xr.Dataset:
        """Open, rename, and slice a dataset from a file or URL.

        Args:
            config: xarray source configuration.

        Returns:
            Lazily loaded, preprocessed xarray Dataset.
        """
        logger.info("Opening dataset from urlpath: %s", config.urlpath)
        dset = xr.open_dataset(
            config.urlpath,
            engine=config.engine,
            chunks=config.chunks or {},
            **config.open_kwargs,
        )
        logger.debug("After open: %s", _ds_summary(dset))
        dset = self._preprocess(dset, config)
        logger.debug("After preprocess (sel/isel): %s", _ds_summary(dset))
        return dset

    def _preprocess(self, dset: xr.Dataset, config: _BaseSourceConfig) -> xr.Dataset:
        """Apply variable renaming, label selection, and index selection from config."""
        mapping = {k: v for k, v in config.mapping.items() if k in dset}
        if mapping:
            logger.debug("Renaming variables: %s", mapping)
            dset = dset.rename(mapping)

        if config.sel:
            sel_kwargs = {k: _parse_sel_value(v) for k, v in config.sel.items()}
            logger.debug("Applying sel: %s", sel_kwargs)
            dset = dset.sel(**sel_kwargs)

        if config.isel:
            isel_kwargs = {k: _parse_sel_value(v) for k, v in config.isel.items()}
            logger.debug("Applying isel: %s", isel_kwargs)
            dset = dset.isel(**isel_kwargs)

        for dim, arr in dset.coords.items():
            if arr.size == 0:
                raise ValueError(
                    f"Selection produced an empty coordinate '{dim}'. "
                    "Check your sel/isel values."
                )

        return dset
