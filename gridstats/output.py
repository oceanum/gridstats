"""Output writers and dataset finalisation for gridstats."""
from __future__ import annotations

import copy
import datetime
import logging
import operator as _op
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from gridstats.config import MaskConfig

import numpy as np
import yaml
import xarray as xr

logger = logging.getLogger(__name__)

_FILLVALUE_ZARR = np.int32(2**31 - 1)
_FILLVALUE_NC = -32767

with open(Path(__file__).parent / "attributes.yml") as _f:
    _METADATA: dict = yaml.safe_load(_f)


# ---------------------------------------------------------------------------
# Attribute helpers
# ---------------------------------------------------------------------------

def set_variable_attributes(dsout: xr.Dataset, extra_metadata: dict[str, Any] = {}) -> xr.Dataset:
    """Apply CF-convention attributes to all variables and coordinates.

    Attribute definitions are loaded from ``attributes.yml``. The ``extra_metadata``
    dict can override or extend entries under the 'coords', 'data_vars', and
    'stats' keys.

    Args:
        dsout: Dataset to annotate.
        extra_metadata: Optional overrides keyed by 'coords', 'data_vars', 'stats'.

    Returns:
        The same dataset with attributes set in-place.
    """
    metadata = copy.deepcopy(_METADATA)
    for key in ("coords", "data_vars", "stats"):
        if key in extra_metadata:
            metadata.setdefault(key, {}).update(extra_metadata[key])

    # Data variables — variable name is expected to be "{base_var}_{stat}"
    for v, dvar in dsout.data_vars.items():
        parts = v.split("_")
        var_name = "_".join(parts[:-1])
        stat_name = parts[-1]
        try:
            stat = metadata["stats"][stat_name]
            attrs = copy.deepcopy(metadata["data_vars"][var_name])
            std = attrs["standard_name"]
            lng = attrs.get("long_name", std.replace("_", " "))
            attrs["standard_name"] = f"{std}_{stat}"
            attrs["long_name"] = f"{stat} {lng}"
            for group_tag in ("month", "year", "season"):
                if group_tag in v:
                    attrs["long_name"] = f"{group_tag}ly " + attrs["long_name"]
                    break
            dvar.attrs = attrs
        except KeyError:
            logger.debug("No metadata found for variable '%s' in attributes.yml", v)

    # Coordinates
    for coord, da in dsout.coords.items():
        if all(a in da.attrs for a in ("standard_name", "long_name", "units")):
            continue
        try:
            attrs = copy.deepcopy(metadata["coords"][coord])
            da.attrs = attrs
        except KeyError:
            logger.debug("No metadata found for coordinate '%s' in attributes.yml", coord)

    return dsout


def set_global_attributes(
    source_ds: xr.Dataset,
    dsout: xr.Dataset,
    extra_attrs: dict[str, Any] = {},
) -> xr.Dataset:
    """Set global CF attributes on the output dataset.

    Defaults are computed from the source dataset; ``extra_attrs`` is merged on
    top so that user-supplied values override or extend the defaults.

    Args:
        source_ds: Original source dataset (used to extract time coverage).
        dsout: Output dataset to annotate.
        extra_attrs: Additional or override global attributes from config.

    Returns:
        The output dataset with global attrs set.
    """
    attrs = {
        "title": "Data stats",
        "institution": "Oceanum",
        "source": "gridstats",
        "date_created": f"{datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d}",
    }
    if "time" in source_ds:
        try:
            t0, t1, tend = source_ds.time[[0, 1, -1]].to_index()
            attrs["time_coverage_start"] = f"{t0:%Y-%m-%d %Hz}"
            attrs["time_coverage_end"] = f"{tend:%Y-%m-%d %Hz}"
            attrs["time_coverage_duration"] = (tend - t0).isoformat()
            attrs["time_coverage_resolution"] = (t1 - t0).isoformat()
        except Exception:
            logger.debug("Could not compute time coverage attributes.")
    attrs.update(extra_attrs)
    dsout.attrs = attrs
    return dsout


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

_OPERATORS = {"gt": _op.gt, "lt": _op.lt, "ge": _op.ge, "le": _op.le}


def _build_mask(source_ds: xr.Dataset, mask_config: "MaskConfig") -> xr.DataArray:
    arr = source_ds[mask_config.var]
    if mask_config.isel:
        arr = arr.isel(mask_config.isel)
    if mask_config.type == "notnull":
        return arr.notnull()
    return _OPERATORS[mask_config.operator](arr, mask_config.value)


# ---------------------------------------------------------------------------
# Finalisation
# ---------------------------------------------------------------------------

def finalise(
    dsout: xr.Dataset,
    source_ds: xr.Dataset,
    chunks: dict[str, int] = {},
    metadata: dict[str, Any] = {},
    global_attrs: dict[str, Any] = {},
    mask_config: "MaskConfig | None" = None,
) -> xr.Dataset:
    """Sort, chunk, transpose, and annotate the output dataset.

    Args:
        dsout: Dataset to finalise.
        source_ds: Original source dataset for global attribute extraction.
        chunks: Output chunking specification.
        metadata: Extra metadata to merge into variable attributes.
        global_attrs: Extra or override global dataset attributes.

    Returns:
        Finalised dataset ready for writing.
    """
    # Sort all dimensions ascending
    for dim in dsout.dims:
        try:
            if dsout[dim][0] > dsout[dim][-1]:
                dsout = dsout.sortby(dim)
        except Exception:
            pass

    # Apply output chunking
    if chunks:
        active_chunks = {k: v for k, v in chunks.items() if k in dsout.dims}
        dsout = dsout.chunk(active_chunks)
        for dvar in dsout.values():
            dvar.encoding.pop("chunks", None)
        for coord in dsout.coords.values():
            coord.encoding.pop("chunks", None)

    # Ensure quantile is first dimension when present
    if "quantile" in dsout.coords:
        dsout = dsout.transpose("quantile", ...)

    # Fix dtypes
    if "season" in dsout.coords:
        dsout["season"] = dsout.season.astype("U")
    for varname in dsout.data_vars:
        if dsout[varname].dtype == "float64":
            dsout[varname] = dsout[varname].astype("float32")

    # Apply spatial mask
    if mask_config is not None:
        mask = _build_mask(source_ds, mask_config)
        dsout = dsout.where(mask)

    # Set attributes
    dsout = set_variable_attributes(dsout, metadata)
    dsout = set_global_attributes(source_ds, dsout, extra_attrs=global_attrs)

    return dsout


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_netcdf(
    dsout: xr.Dataset,
    path: str,
    fill_value: int = _FILLVALUE_NC,
    format: str = "NETCDF4",
) -> None:
    """Write the dataset to a NetCDF file with zlib compression.

    Args:
        dsout: Dataset to write.
        path: Output file path.
        fill_value: Fill value for all data variables.
        format: NetCDF format string.
    """
    logger.info("Writing NetCDF: %s", Path(path).absolute())
    encoding = {v: {"zlib": True, "_FillValue": fill_value} for v in dsout.data_vars}
    dsout.to_netcdf(path, format=format, encoding=encoding)


def write_zarr(
    dsout: xr.Dataset,
    path: str,
    fill_value: int = int(_FILLVALUE_ZARR),
    append: bool = False,
    consolidate: bool = False,
    **kwargs,
) -> None:
    """Write the dataset to a Zarr store.

    Args:
        dsout: Dataset to write.
        path: Output Zarr store path or URL.
        fill_value: Fill value applied to all data variables.
        append: If True, add variables to an existing store (creating it if needed)
            rather than overwriting. Variables that already exist are deleted and
            rewritten. Coordinates shared across parallel tasks are handled safely
            via zarr's ``require_dataset``. Consolidated metadata is intentionally
            skipped so that parallel writers do not race on ``.zmetadata`` — use
            ``consolidate=True`` in a final dependent task to produce it.
        consolidate: If True, call ``zarr.consolidate_metadata`` after writing.
            Use this on the final task that depends on all parallel writers.
        **kwargs: Forwarded to ``Dataset.to_zarr``.
    """
    for varname in dsout.data_vars:
        dsout[varname].encoding.update({"_FillValue": fill_value})
        dsout[varname].encoding.pop("zlib", None)

    if append:
        import zarr

        logger.info("Writing Zarr (append): %s", path)
        store = zarr.open_group(path, mode="a")
        for var in dsout.data_vars:
            if var in store:
                logger.info("Removing existing variable '%s' before overwrite", var)
                del store[var]
        dsout.to_zarr(path, mode="a", consolidated=False, **kwargs)
    else:
        logger.info("Writing Zarr: %s", path)
        dsout.to_zarr(path, mode="w", **kwargs)

    if consolidate:
        import zarr

        logger.info("Consolidating Zarr metadata: %s", path)
        zarr.consolidate_metadata(path)


def write(
    dsout: xr.Dataset,
    path: str,
    append: bool = False,
    consolidate: bool = False,
    **kwargs,
) -> None:
    """Dispatch to write_netcdf or write_zarr based on the file extension.

    Args:
        dsout: Dataset to write.
        path: Output path. Must end in '.nc' or '.zarr'.
        append: Passed to ``write_zarr`` (ignored for NetCDF).
        consolidate: Passed to ``write_zarr`` (ignored for NetCDF).
        **kwargs: Forwarded to the chosen writer.

    Raises:
        ValueError: If the extension is not '.nc' or '.zarr'.
    """
    if path.endswith(".nc"):
        write_netcdf(dsout, path, **kwargs)
    elif path.endswith(".zarr"):
        write_zarr(dsout, path, append=append, consolidate=consolidate, **kwargs)
    else:
        raise ValueError(
            f"Output path must end with '.nc' or '.zarr', got: {path!r}"
        )
