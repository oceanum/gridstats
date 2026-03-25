"""Pipeline orchestration for gridstats.

The Pipeline class loads data, applies a sequence of stat operations defined
in a PipelineConfig, finalises the result, and writes it to disk.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import dask
import numpy as np
import xarray as xr

from gridstats.config import CallConfig, PipelineConfig
from gridstats.output import finalise, write
from gridstats.registry import get_loader, get_stat

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dask cluster helpers
# ---------------------------------------------------------------------------

@contextmanager
def _dummy_client(**kwargs) -> Generator:
    """No-op context manager used when Dask cluster is disabled."""
    yield None


@contextmanager
def _local_cluster(**kwargs) -> Generator:
    """Start a local Dask distributed cluster and yield the client."""
    from dask.distributed import Client

    with Client(**kwargs) as client:
        logger.info("Dask cluster started: %s", client.dashboard_link)
        yield client


# ---------------------------------------------------------------------------
# Spatial tiling (replaces @stepwise decorator from old utils.py)
# ---------------------------------------------------------------------------

def _apply_tiled(
    fn,
    data: xr.Dataset,
    tiles: dict[str, int],
    **kwargs,
) -> xr.Dataset:
    """Apply fn to spatial tiles of data and reassemble with combine_by_coords.

    Args:
        fn: Stat function with signature fn(data, **kwargs) -> xr.Dataset.
        data: Full dataset.
        tiles: Mapping of dimension name → tile size (e.g. {'latitude': 20}).
        **kwargs: Forwarded to fn.

    Returns:
        Reassembled result dataset.
    """
    import pandas as pd

    tile_dims = {k: v for k, v in tiles.items() if k in data.dims}
    if not tile_dims:
        return fn(data, **kwargs)

    # Build index slices for each tile dimension
    dim_slices: dict[str, list[slice]] = {}
    for dim, step in tile_dims.items():
        size = data.sizes[dim]
        dim_slices[dim] = [
            slice(i, min(i + step, size)) for i in range(0, size, step)
        ]

    # Iterate over all tile combinations
    from itertools import product

    dims = list(dim_slices)
    all_slices = list(product(*[dim_slices[d] for d in dims]))

    results = []
    for i, slices in enumerate(all_slices):
        logger.info("Tile %d/%d", i + 1, len(all_slices))
        isel_kwargs = {d: s for d, s in zip(dims, slices)}
        tile = data.isel(isel_kwargs)
        results.append(fn(tile, **kwargs))

    return xr.combine_by_coords(results)


# ---------------------------------------------------------------------------
# Directional sectorisation (ported from stats.py._directional_stat)
# ---------------------------------------------------------------------------

def _apply_directional(
    fn,
    data: xr.Dataset,
    dir_var: str,
    nsector: int,
    **kwargs,
) -> xr.Dataset:
    """Bin data by directional sectors and apply fn to each sector.

    Args:
        fn: Stat function with signature fn(data, **kwargs) -> xr.Dataset.
        data: Full dataset including the directional variable.
        dir_var: Name of the direction variable used for binning.
        nsector: Number of equal sectors covering 360°.
        **kwargs: Forwarded to fn (should not include dir_var).

    Returns:
        Dataset with a new 'direction' dimension containing sector centres.
    """
    if dir_var not in data:
        raise ValueError(
            f"Directional variable '{dir_var}' not found in dataset."
        )

    dirs = data[dir_var]
    dsector = 360.0 / nsector
    sector_centres = np.linspace(0, 360 - dsector, nsector)
    starts = (sector_centres - dsector / 2) % 360
    stops = (sector_centres + dsector / 2) % 360

    results = []
    for start, stop in zip(starts, stops):
        logger.info("Directional sector [%.1f, %.1f)", start, stop)
        mask = (dirs >= start) & (dirs < stop) if stop > start else (dirs >= start) | (dirs < stop)
        results.append(fn(data.where(mask), **kwargs))

    dsout = xr.concat(results, dim="direction").assign_coords(
        {"direction": sector_centres}
    )
    dsout["direction"].attrs = {
        "standard_name": dirs.attrs.get("standard_name", "direction") + "_sector",
        "long_name": dirs.attrs.get("long_name", "direction") + " sector",
        "units": dirs.attrs.get("units", "degree"),
    }
    return dsout


# ---------------------------------------------------------------------------
# Loader selection
# ---------------------------------------------------------------------------

def _select_loader(config):
    """Return the appropriate loader instance for the given source config."""
    return get_loader(config.type)()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """Orchestrate data loading, stat computation, and output writing.

    Args:
        config: Validated pipeline configuration.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        # Match old onstats behaviour: do not auto-split large chunks during
        # rechunking or groupby operations.  Without this, newer dask versions
        # default to split_large_chunks=True and may create unexpectedly large
        # intermediate arrays when rechunking a time=-1 dataset for grouped
        # quantile computations.
        dask.config.set({"array.slicing.split_large_chunks": False})

    @classmethod
    def from_yaml(cls, path: str | Path) -> Pipeline:
        """Instantiate a Pipeline from a YAML configuration file.

        Args:
            path: Path to a YAML file conforming to the PipelineConfig schema.

        Returns:
            Ready-to-run Pipeline instance.
        """
        return cls(PipelineConfig.from_yaml(path))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> xr.Dataset:
        """Execute the full pipeline: load → compute → finalise → write.

        Returns:
            The finalised output dataset (also written to disk).
        """
        logger.info("Pipeline starting.")

        dsout = xr.Dataset()
        for call in self.config.calls:
            logger.info("Applying stat: %s", call.func)
            result = self._apply(call)
            dsout = dsout.merge(result)

        source_ds = self._load()

        dsout = finalise(
            dsout,
            source_ds,
            chunks=self.config.output.__dict__.get("chunks", {}),
            metadata=self.config.metadata,
        )

        write(dsout, self.config.output.outfile)
        logger.info("Pipeline complete. Output: %s", self.config.output.outfile)
        return dsout

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, chunks: dict[str, int] | None = None) -> xr.Dataset:
        """Load the source dataset, optionally overriding chunk sizes.

        Args:
            chunks: Chunk sizes to use when opening the dataset. Merged on top of
                any chunks defined in the source config; call-level chunks take
                priority.
        """
        if self.config.source is None:
            raise NotImplementedError(
                "Multi-source pipelines (config.sources) are not yet implemented. "
                "Use config.source for a single source."
            )
        source = self.config.source
        if chunks:
            source = source.model_copy(update={"chunks": {**source.chunks, **chunks}})
        loader = _select_loader(source)
        return loader.load(source)

    def _apply(self, call: CallConfig) -> xr.Dataset:
        """Run a single stat call and return the renamed result dataset."""
        fn = get_stat(call.func)

        # Load with source-level chunks only (native zarr chunks), then select
        # the required variables before applying call-level rechunking.  This
        # avoids creating rechunk tasks for variables that are not needed by
        # this call, reducing peak dask graph size and intermediate memory.
        from gridstats.loaders.xarray import _ds_summary

        data = self._load()
        logger.info("[%s] After load: %s", call.func, _ds_summary(data))

        # --- Variable selection (before rechunking) ---
        if call.data_vars != "all":
            data = data[call.data_vars]
            logger.info("[%s] After var selection %s: %s", call.func, call.data_vars, _ds_summary(data))

        # --- Apply call-level rechunking to selected variables only ---
        if call.chunks:
            data = data.chunk(call.chunks)
            logger.info("[%s] After call-level chunk %s: %s", call.func, call.chunks, _ds_summary(data))

        # --- Build kwargs for the stat function ---
        fn_kwargs = dict(
            dim=call.dim,
            group=call.group,
            **call.extra_kwargs(),
        )

        # --- Dask cluster ---
        cluster_ctx = (
            _local_cluster(
                **{
                    k: v
                    for k, v in self.config.cluster.model_dump(exclude={"enabled"}).items()
                    if v is not None
                }
            )
            if self.config.cluster.enabled and call.use_dask_cluster
            else _dummy_client()
        )

        with cluster_ctx:
            # --- Directional sectorisation ---
            if call.nsector is not None:
                result = _apply_directional(
                    lambda d, **kw: fn(d, **kw),
                    data,
                    dir_var=call.dir_var,
                    nsector=call.nsector,
                    **fn_kwargs,
                )
            # --- Spatial tiling ---
            elif call.tiles:
                result = _apply_tiled(fn, data, call.tiles, **fn_kwargs)
            else:
                result = fn(data, **fn_kwargs)

            logger.info("[%s] Triggering compute (result.load()): %s", call.func, _ds_summary(result))
            result = result.load()
            logger.info("[%s] Compute complete: %s", call.func, _ds_summary(result))

        # --- Rename output variables with suffix ---
        suffix = call.suffix
        if suffix is None:
            suffix = f"_{call.func}"
            if call.group:
                suffix += f"_{call.group}"
            if call.nsector is not None:
                suffix += "_direc"

        return result.rename({v: f"{v}{suffix}" for v in result.data_vars})
