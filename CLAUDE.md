# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**onstats** is an Oceanum library for computing statistical summaries and derived variables from oceanographic datasets (wave, ocean, wind data). It uses xarray and dask for processing large gridded datasets.

## Commands

```bash
# Install for development
pip install -e .

# Run tests
make test              # Quick pytest run
pytest tests/          # Equivalent
pytest tests/test_cli.py::test_name  # Single test

# Lint
make lint              # flake8 onstats tests
flake8 onstats tests   # Equivalent

# Coverage
make coverage

# Build distribution
make dist

# Clean artifacts
make clean
```

## Architecture

### Plugin System

The core design is a **plugin-based architecture** in [onstats/stats.py](onstats/stats.py). The `Stats` class uses a `Plugin` metaclass that automatically discovers and attaches all public functions from the `onstats/functions/` subpackage as methods at runtime. Functions prefixed with `_` are private and not attached.

Each plugin function signature must be: `func(self, dset, **kwargs)` where `dset` is an xarray Dataset.

### Key Classes

- **`Stats`** ([onstats/stats.py](onstats/stats.py)): Main entry point. Loads datasets (via xarray, intake catalog, or urlpath), sets up dask clusters for parallel processing, and executes a sequence of operations defined in the `calls` config list. Outputs NetCDF or Zarr.

- **`DerivedVariable`** ([onstats/derived_variable.py](onstats/derived_variable.py)): Computes derived oceanographic variables (e.g., Douglas sea scale, wind speed from components) from raw dataset variables. Used inside Stats for variable mapping.

- **`KMZ`** ([onstats/kmzstats_new.py](onstats/kmzstats_new.py)): Generates KML/KMZ files for Google Earth visualization from statistical output. Driven by YAML config.

### Functions Subpackage ([onstats/functions/](onstats/functions/))

Pluggable statistical operations auto-attached to `Stats`:
- `distribution.py` — joint 2D/3D distributions
- `exceedance.py` — probability of exceedance
- `rpv.py` — return period values (extreme value analysis)
- `frequency_domain.py` — wave spectral analysis
- `directional.py` — directional statistics
- `probability.py` — probability distributions
- `windpower.py` — wind power calculations

### Computation Stack

- **xarray** — multi-dimensional data operations ([onstats/xarray_stats.py](onstats/xarray_stats.py) wraps NumPy with dask support)
- **NumPy/SciPy** — low-level implementations ([onstats/numpy_stats.py](onstats/numpy_stats.py): peaks-over-threshold, histograms, return period values)
- **dask/distributed** — parallel and out-of-core computation
- **intake-forecast** — data catalog access

### Configuration

Operations are driven by YAML config files. See [onstats/stats.yml](onstats/stats.yml) for the template and [onstats/README.rst](onstats/README.rst) for examples. Key fields:
- Data source: `dset`, `urlpath`, or `catalog` + `dataset_id`
- Variable renaming: `mapping`
- Output: `outfile`
- Operations sequence: `calls` list

Variable metadata (units, long names, etc.) is defined in [onstats/attributes.yml](onstats/attributes.yml).

## CLI

```bash
# Run gridded statistics from YAML config
onstats gridstats config.yml

# Generate KMZ visualization
onstats kmz config.yml -o ./output -k output.kmz
```
