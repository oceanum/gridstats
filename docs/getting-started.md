# Getting Started

## Installation

```bash
pip install onstats
```

Requires Python ≥ 3.10.

For intake catalog support (loading data from an intake catalog):

```bash
pip install "onstats[intake]"
```

## How it works

A pipeline has three parts:

1. **Source** — where to load the data from (a local or remote file, or an intake catalog)
2. **Calls** — an ordered list of stat operations to compute
3. **Output** — where to write the results (NetCDF or Zarr)

Each call names a registered stat function (`func`) and passes parameters to it. The pipeline runs each call in sequence, accumulates results, and writes a single output dataset.

## Your first pipeline

### 1. Prepare a config file

```yaml title="stats.yml"
source:
  type: xarray
  urlpath: /data/hindcast/waves.zarr
  engine: zarr

output:
  outfile: ./wave_stats.nc

calls:
  - func: mean
    dim: time
    data_vars: [hs, tp]

  - func: quantile
    dim: time
    data_vars: [hs]
    q: [0.5, 0.95, 0.99]
```

### 2. Run it

```bash
onstats run stats.yml
```

The output file `wave_stats.nc` will contain:

| Variable | Description |
|---|---|
| `hs_mean` | Time-mean significant wave height |
| `tp_mean` | Time-mean peak period |
| `hs_quantile` | Hs at the 50th, 95th and 99th percentiles |

All variables carry CF-convention attributes (standard name, long name, units).

### 3. Load the result

```python
import xarray as xr

ds = xr.open_dataset("wave_stats.nc")
print(ds)
```

## Grouping by time

Add `group` to any call to compute stats per calendar period:

```yaml
calls:
  - func: mean
    dim: time
    data_vars: [hs]
    group: month      # one value per month (1–12)

  - func: max
    dim: time
    data_vars: [hs]
    group: season     # one value per season (DJF, MAM, JJA, SON)
```

## Variable renaming

Use `mapping` to rename input variables before processing:

```yaml
source:
  type: xarray
  urlpath: /data/model.zarr
  engine: zarr
  mapping:
    tps: tp      # model uses 'tps', rename to 'tp'
    hs0: hs
```

## Selecting variables and slicing

Restrict to specific variables per call with `data_vars`. Use `sel` on the source to crop the spatial or temporal domain before any statistics are computed:

```yaml
source:
  type: xarray
  urlpath: /data/hindcast.zarr
  engine: zarr
  sel:
    latitude: {start: -50, stop: -30}
    longitude: {start: 160, stop: 180}
    time: {start: "2000-01-01", stop: "2020-12-31"}

calls:
  - func: mean
    dim: time
    data_vars: [hs]   # only process 'hs'
```

## Memory management

For datasets that are too large to rechunk fully into memory, use `tiles` to process in spatial blocks:

```yaml
calls:
  - func: quantile
    dim: time
    data_vars: [hs]
    q: [0.95]
    chunks:
      time: -1          # quantile needs the full time axis in one chunk
      latitude: 50
      longitude: 50
    tiles:
      latitude: 10      # process 10 rows at a time
      longitude: 10
```

## Using a Dask cluster

For very large datasets enable the distributed Dask cluster:

```yaml
cluster:
  enabled: true
  n_workers: 4
  threads_per_worker: 2

calls:
  - func: rpv
    dim: time
    data_vars: [hs]
    return_periods: [10, 100]
    use_dask_cluster: true
```

`use_dask_cluster: false` on a specific call lets you bypass the cluster for cheap operations.

## CLI reference

```
Usage: onstats [OPTIONS] COMMAND [ARGS]...

  Compute gridded statistics on oceanographic datasets.

Commands:
  run         Run a stats pipeline from a YAML configuration file.
  list-stats  List all registered stat functions.
```
