# gridstats

**gridstats** is an Oceanum library for computing gridded statistics over large oceanographic and climate datasets. Pipelines are defined in YAML and run as a single CLI command. Computation is lazy and out-of-core via [xarray](https://xarray.dev) and [dask](https://dask.org), so datasets of arbitrary size are handled without loading them into memory.

## Features

- **YAML-driven pipelines** — source, operations, and output are all declared in one config file
- **Out-of-core** — processes arbitrarily large grids lazily; spatial `tiles:` keeps peak memory bounded
- **Rich stat library** — aggregations, quantiles, exceedance, return period values, directional stats, distributions, and more
- **Multiple output formats** — NetCDF or Zarr
- **CF-compliant** — output variables are automatically annotated with standard names, units, and long names
- **Extensible** — register custom stat functions and loaders via decorator or package entry point

## Installation

Requires Python ≥ 3.10.

```bash
pip install gridstats
```

For loading data from an [intake](https://intake.readthedocs.io) catalog:

```bash
pip install "gridstats[intake]"
```

## Quick start

### 1. Write a config file

```yaml
# stats.yml
source:
  type: xarray
  urlpath: /data/hindcast/waves.zarr
  engine: zarr
  sel:
    time: {start: "2000-01-01", stop: "2020-12-31"}
    latitude: {start: -50, stop: -30}
    longitude: {start: 160, stop: 180}

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
```

### 2. Run it

```bash
gridstats run stats.yml
```

The output dataset will contain variables like `hs_mean`, `tp_mean`, `hs_quantile`, and `hs_rpv`, each with CF-standard attributes.

### 3. Use the result

```python
import xarray as xr

ds = xr.open_zarr("wave_stats.zarr")
print(ds)
```

## Available stat functions

| Function | Description |
|---|---|
| `mean`, `max`, `min`, `std`, `count` | Basic aggregations |
| `quantile` | Quantiles at arbitrary probability levels |
| `pcount` | Count of non-NaN values per grid cell |
| `exceedance` / `nonexceedance` | Probability of exceeding a threshold |
| `range_probability` | Probability of a value falling in a range |
| `rpv` | Return period values via extreme value fitting |
| `distribution2` / `distribution3` | 2- and 3-parameter distribution fitting |
| `statdir` | Directional statistics (sector-binned) |
| `hmo` | Significant wave height from spectral moments |
| `winpow` | Wind power density |

All calls accept a `group:` key (`month`, `season`, `hour`, …) to compute statistics per calendar period.

## Grouping and spatial tiling

```yaml
calls:
  # Monthly mean
  - func: mean
    dim: time
    data_vars: [hs]
    group: month

  # Quantile with spatial tiling to control memory
  - func: quantile
    dim: time
    data_vars: [hs]
    q: [0.95, 0.99]
    chunks: {time: -1, latitude: 50, longitude: 50}
    tiles: {latitude: 10, longitude: 10}
```

## Plugin system

Register a custom stat function in your own package:

```python
from gridstats.registry import register_stat
import xarray as xr

@register_stat("my_stat")
def my_stat(data: xr.Dataset, *, dim: str = "time", **kwargs) -> xr.Dataset:
    ...
```

Or declare it as a package entry point so it is discovered automatically:

```toml
[project.entry-points."gridstats.stats"]
my_stat = "my_package.stats:my_stat"
```

## CLI

```
Usage: gridstats [OPTIONS] COMMAND [ARGS]...

Commands:
  run         Run a stats pipeline from a YAML configuration file.
  list-stats  List all registered stat functions.
```

## License

MIT — see [LICENSE](LICENSE).
