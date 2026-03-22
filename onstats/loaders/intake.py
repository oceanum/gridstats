"""Intake catalog data loader."""
from __future__ import annotations

import logging

import xarray as xr

from onstats.config import IntakeSourceConfig
from onstats.registry import register_loader

logger = logging.getLogger(__name__)


@register_loader("intake")
class IntakeLoader:
    """Load datasets from an intake-forecast catalog."""

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
                "intake-forecast is required for catalog loading: "
                "pip install intake-forecast"
            ) from exc

        logger.info(
            "Opening dataset from catalog: %s[%s]",
            config.catalog,
            config.dataset_id,
        )
        cat = open_catalog(config.catalog)
        dset = cat[config.dataset_id].to_dask()
        if config.chunks:
            dset = dset.chunk(config.chunks)

        from onstats.loaders.xarray import XarrayLoader

        return XarrayLoader()._preprocess(dset, config)
