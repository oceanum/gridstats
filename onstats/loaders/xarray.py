"""xarray-based data loader."""
from __future__ import annotations

import logging

import xarray as xr

from onstats.config import XarraySourceConfig, _BaseSourceConfig
from onstats.registry import register_loader

logger = logging.getLogger(__name__)


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
        )
        return self._preprocess(dset, config)

    def _preprocess(self, dset: xr.Dataset, config: _BaseSourceConfig) -> xr.Dataset:
        """Apply variable renaming and slicing from config."""
        mapping = {k: v for k, v in config.mapping.items() if k in dset}
        if mapping:
            logger.debug("Renaming variables: %s", mapping)
            dset = dset.rename(mapping)

        for method, kwargs in config.slice_dict.items():
            dset = getattr(dset, method)(**kwargs)
            for coord, arr in dset.coords.items():
                if arr.size == 0:
                    raise ValueError(
                        f"Slicing with {method}({kwargs}) produced empty "
                        f"coordinate '{coord}'"
                    )

        return dset
