"""Intake catalog data loader."""
from __future__ import annotations

import logging

import xarray as xr

from gridstats.config import IntakeSourceConfig
from gridstats.loaders.xarray import XarrayLoader, _ds_summary
from gridstats.registry import register_loader

logger = logging.getLogger(__name__)


@register_loader("intake")
class IntakeLoader:
    """Load datasets from an intake catalog."""

    def load(self, config: IntakeSourceConfig) -> xr.Dataset:
        """Open, rename, and slice a dataset from an intake catalog.

        Args:
            config: Intake source configuration.

        Returns:
            Lazily loaded, preprocessed xarray Dataset.

        Raises:
            ImportError: If intake-forecast is not installed.
        """
        try:
            from intake import open_catalog
        except ImportError as exc:
            raise ImportError(
                "intake is required for catalog loading: "
                "pip install gridstats[intake]"
            ) from exc

        logger.info(
            "Opening dataset from catalog: %s[%s]",
            config.catalog,
            config.dataset_id,
        )
        cat = open_catalog(config.catalog)
        dset = cat[config.dataset_id].to_dask()
        logger.debug("After to_dask: %s", _ds_summary(dset))
        dset = XarrayLoader()._preprocess(dset, config)
        logger.debug("After preprocess (sel/isel): %s", _ds_summary(dset))
        if config.chunks:
            dset = dset.chunk(config.chunks)
            logger.debug("After source-level chunk: %s", _ds_summary(dset))
        return dset
