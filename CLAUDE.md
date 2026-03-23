# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**gridstats** is an Oceanum library for computing statistical summaries from oceanographic datasets (wave, wind, current, temperature). It processes large gridded datasets lazily with xarray + dask and writes results to NetCDF or Zarr.

## Commands

All commands should be run inside the `gridstats` virtual environment:

```bash
source ~/.virtualenvs/gridstats/bin/activate

# Install for development
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_ops.py::test_mean

# Build docs
mkdocs build
mkdocs serve        # live preview at http://127.0.0.1:8000

# CLI
gridstats run config.yml
gridstats list-stats
```

## Architecture

### Key modules

| Module | Role |
|---|---|
| `gridstats/config.py` | Pydantic v2 models for all YAML config fields |
| `gridstats/registry.py` | Dict-based registry; `@register_stat` / `@register_loader` decorators |
| `gridstats/pipeline.py` | `Pipeline` class — loads data, runs calls, writes output |
| `gridstats/output.py` | `finalise()`, `write()`, CF attribute application |
| `gridstats/ops/` | Stat functions (one file per category) |
| `gridstats/loaders/` | `XarrayLoader` and `IntakeLoader` |
| `gridstats/cli.py` | typer CLI (`run`, `list-stats`) |

### Config flow

A YAML config is parsed by `PipelineConfig.from_yaml()` into typed Pydantic models. `Pipeline` reads the config, selects a loader from the registry by inspecting which `source` fields are set (`urlpath` → xarray, `catalog+dataset_id` → intake), then iterates over the `calls` list. Each call looks up the function by `func` name in the registry and applies it to the data, optionally with spatial tiling (`tiles:`) or directional sectorisation (`nsector:`). Results are merged into a single output dataset, CF attributes are applied, and the dataset is written to disk.

### Registry and plugins

`_STATS` and `_LOADERS` are plain dicts. Built-in ops register themselves via `@register_stat("name")` when their modules are imported. `gridstats/__init__.py` imports `gridstats.loaders` and `gridstats.ops` to trigger registration, then calls `_load_entrypoint_plugins()` to discover third-party extensions declared under the `gridstats.stats` or `gridstats.loaders` entry-point groups.

### Stat function signature

```python
def my_stat(data: xr.Dataset, *, dim: str = "time", group: str | None = None, **kwargs) -> xr.Dataset:
    ...
```

Extra call-level YAML keys (anything not in `CallConfig`'s explicit fields) are collected by `CallConfig.extra_kwargs()` and forwarded as `**kwargs`.

### Spatial tiling

When `tiles:` is set on a call, `_apply_tiled()` in `pipeline.py` iterates over lat/lon blocks, calls the stat function on each block, and `xr.combine_by_coords()` reassembles the result. This keeps peak memory bounded for large grids.

### Output naming

Each output variable is renamed with a suffix before merging:

```
{variable}_{func}[_{group}][_direc]
```

Override with `suffix:` on the call. The `exceedance` family appends the threshold value instead.

### Variable metadata

`gridstats/attributes.yml` maps variable names to CF standard attributes (standard_name, long_name, units). `output.set_variable_attributes()` applies these after all calls complete.

### Multi-source (placeholder)

`PipelineConfig.sources` (dict) is accepted by the schema but raises `NotImplementedError` at runtime. Only single-source pipelines (`source:`) are currently implemented.
