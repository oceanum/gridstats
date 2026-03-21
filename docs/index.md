# onstats

**onstats** is an Oceanum library for computing gridded statistics over large oceanographic and climate datasets. It is built on [xarray](https://xarray.dev) and [dask](https://dask.org) for lazy, out-of-core computation and is driven entirely by YAML configuration files.

## Key features

- **YAML-driven** — define your full pipeline in a config file, run it with one command
- **Out-of-core** — processes arbitrarily large datasets via dask without loading into memory
- **Extensible** — register custom stat functions and loaders with a simple decorator or entry point
- **Multiple output formats** — write results to NetCDF or Zarr
- **CF-compliant metadata** — output variables are automatically annotated with standard names, units and long names

## Quick example

Write a config file:

```yaml title="config.yml"
source:
  urlpath: gs://my-bucket/hindcast.zarr
  engine: zarr
  mapping:
    tps: tp        # rename variables on load

output:
  outfile: ./wave_stats.zarr

calls:
  - func: mean
    dim: time
    data_vars: [hs, tp]

  - func: quantile
    dim: time
    data_vars: [hs]
    q: [0.5, 0.90, 0.95, 0.99]

  - func: rpv
    dim: time
    data_vars: [hs]
    return_periods: [10, 50, 100]
    distribution: gumbel_r
    duration: 24
```

Run it:

```bash
onstats run config.yml
```

## Installation

```bash
pip install onstats
```

To include intake catalog support:

```bash
pip install "onstats[intake]"
```

## Next steps

- [Getting Started](getting-started.md) — a complete walkthrough
- [Configuration](configuration.md) — full YAML schema reference
- [Operations](ops/index.md) — all built-in stat functions
- [Custom Plugins](plugins.md) — add your own loaders and stats
